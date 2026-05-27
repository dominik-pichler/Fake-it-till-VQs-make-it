"""
Multi-encoder feature extractor with learned weighted-sum fusion.

Three frozen pre-trained backbones produce embeddings from the same image,
each is projected to a common dimension, and the three projected vectors
are combined via a learned weighted sum (softmax-normalised weights).

    1. SL  -- ResNet101 pre-trained on ImageNet (supervised classification).
              Encoder head = global-pooled features after the final conv block.
    2. VSL -- ViT-Base/16 pre-trained on ImageNet (supervised classification).
              We extract the class-token representation.
    3. SSL -- DINO ResNet50 (self-supervised, no labels).
              Encoder head = global-pooled features.

Designed to slot next to a hand-crafted spectral feature extractor: this
module returns a 1-D torch tensor (or numpy array) per image that can be
concatenated downstream.

Usage
-----
    extractor = MultiEncoderExtractor(embed_dim=512, device="cuda")
    vec = extractor.extract(pil_image_or_path_or_tensor)   # -> (embed_dim,) tensor
    batch = extractor.extract_batch([img1, img2, ...])     # -> (N, embed_dim)

Notes
-----
- Backbones are frozen (eval mode, requires_grad=False). Only the three
  projection heads and the three fusion weights are trainable.
- Inputs are normalised to 256x256 to match the spectral pipeline. The
  ResNets handle this size natively via global pooling; the ViT runs at
  its native 224x224 input (we resize 256 -> 224 inside the encoder so
  the pipeline doesn't need to know).
- ImageNet mean/std normalisation is applied inside the module so callers
  can pass raw [0, 1] images or PIL images directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import torch
torch.backends.mkldnn.enabled = False # Fix as it causes issues with oneDNN (MKLDNN)
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    resnet101, ResNet101_Weights,
    resnet18, ResNet18_Weights,
    vit_b_16, ViT_B_16_Weights,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MultiEncoderConfig:
    image_size: int = 256
    embed_dim: int = 512           # common projection dim & final output dim
    use_sl: bool = True            # ResNet101 (ImageNet supervised)
    use_vsl: bool = True           # ViT-B/16 (ImageNet supervised)
    use_ssl: bool = True           # DINO ResNet50 (self-supervised)
    freeze_backbones: bool = True
    # If True, replace the three-backbone stack with a single ResNet18.
    # 11M params and ~20x faster than ResNet101+ViT+DINO on CPU; meant for
    # boxes without a GPU. The use_sl/use_vsl/use_ssl flags are ignored
    # when lightweight is set.
    lightweight: bool = False


# ImageNet statistics used by all three backbones.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Backbone wrappers
# ---------------------------------------------------------------------------

class _ResNet101Encoder(nn.Module):
    """ResNet101 pre-trained on ImageNet, with the classification head removed."""

    out_dim = 2048

    def __init__(self):
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V2
        net = resnet101(weights=weights)
        # Drop the final FC layer; keep everything up to and including avgpool.
        self.backbone = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)          # (B, 2048, 1, 1)
        return feat.flatten(1)           # (B, 2048)


class _ResNet18Encoder(nn.Module):
    """ResNet18 pre-trained on ImageNet, with the classification head removed.

    Drop-in CPU-friendly replacement for the three-backbone stack. About
    11M params vs. ResNet101's 45M, and the lighter residual blocks make
    per-image latency ~20x smaller on CPU. Embedding dim is 512 (not 2048),
    which happens to match MultiEncoderConfig.embed_dim's default -- no
    information bottleneck at the projection head.
    """

    out_dim = 512

    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        net = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)          # (B, 512, 1, 1)
        return feat.flatten(1)           # (B, 512)


class _ViTEncoder(nn.Module):
    """
    ViT-Base/16 pre-trained on ImageNet, returning the class-token feature.

    Uses torchvision's vit_b_16 to avoid a timm dependency. torchvision's
    pretrained ViT is fixed at 224x224 (positional embeddings are not
    interpolated automatically), so we resize the input down from whatever
    the rest of the pipeline runs at to 224 before feeding the backbone.
    We then read the CLS-token output by replacing the classification head
    with an Identity.
    """

    out_dim = 768
    _VIT_INPUT_SIZE = 224

    def __init__(self, image_size: int = 256):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        net = vit_b_16(weights=weights)
        # Replace the classification head so forward() returns the 768-dim
        # CLS-token feature (torchvision's ViT applies head AFTER the
        # encoder, and the head's input is the pooled CLS embedding).
        net.heads = nn.Identity()
        self.backbone = net
        # Resize to the ViT's native input only if the upstream pipeline
        # is at a different size. interpolate at forward time so we don't
        # carry a torchvision Resize transform (which can't see grad).
        self._needs_resize = image_size != self._VIT_INPUT_SIZE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._needs_resize:
            x = F.interpolate(
                x, size=(self._VIT_INPUT_SIZE, self._VIT_INPUT_SIZE),
                mode="bicubic", align_corners=False, antialias=True,
            )
        return self.backbone(x)          # (B, 768)


class _DinoResNet50Encoder(nn.Module):
    """DINO self-supervised ResNet50 (loaded from facebookresearch/dino)."""

    out_dim = 2048

    def __init__(self):
        super().__init__()
        # torch.hub fetch; this is the official DINO release.
        self.backbone = torch.hub.load(
            "facebookresearch/dino:main", "dino_resnet50"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)          # (B, 2048)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class MultiEncoderExtractor(nn.Module):
    """
    Three frozen encoders, three projection heads, one learned weighted sum.

    Trainable parameters:
      - One Linear(in_dim -> embed_dim) per active encoder.
      - A length-K vector of fusion logits (K = number of active encoders),
        softmaxed at forward time so the effective weights sum to 1 and stay
        interpretable.
    """

    def __init__(
        self,
        config: MultiEncoderConfig | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.cfg = config or MultiEncoderConfig()
        self.device = torch.device(device)

        # Build the active encoders.
        encoders: dict[str, nn.Module] = {}
        if self.cfg.lightweight:
            # CPU-only escape hatch: a single small backbone. Ignores
            # use_sl/use_vsl/use_ssl on purpose.
            encoders["rn18"] = _ResNet18Encoder()
        else:
            if self.cfg.use_sl:
                encoders["sl"] = _ResNet101Encoder()
            if self.cfg.use_vsl:
                encoders["vsl"] = _ViTEncoder(image_size=self.cfg.image_size)
            if self.cfg.use_ssl:
                encoders["ssl"] = _DinoResNet50Encoder()

        if not encoders:
            raise ValueError("At least one encoder must be enabled.")

        self.encoder_names: list[str] = list(encoders.keys())
        self.encoders = nn.ModuleDict(encoders)

        # One projection head per encoder, all mapping to embed_dim.
        self.projections = nn.ModuleDict({
            name: nn.Linear(enc.out_dim, self.cfg.embed_dim)
            for name, enc in encoders.items()
        })

        # Fusion logits -- softmaxed to non-negative weights summing to 1.
        # Initialised to zero -> uniform weighting at the start.
        self.fusion_logits = nn.Parameter(torch.zeros(len(self.encoder_names)))

        # Freeze backbones if requested. Projection heads & fusion logits
        # remain trainable.
        if self.cfg.freeze_backbones:
            for enc in self.encoders.values():
                for p in enc.parameters():
                    p.requires_grad = False
                enc.eval()

        # Image pre-processing pipeline. Accepts PIL / ndarray / tensor.
        self._preprocess = transforms.Compose([
            transforms.Resize(
                (self.cfg.image_size, self.cfg.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),       # -> float tensor in [0, 1], (C, H, W)
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        self.to(self.device)

    # ----- properties -----

    @property
    def fusion_weights(self) -> torch.Tensor:
        """Current effective weights after softmax (sums to 1)."""
        return F.softmax(self.fusion_logits, dim=0)

    @property
    def n_features(self) -> int:
        return self.cfg.embed_dim

    @property
    def feature_names(self) -> list[str]:
        return [f"deep_{i:04d}" for i in range(self.n_features)]

    # ----- core forward -----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: pre-processed image batch, shape (B, 3, H, W), already normalised
           with ImageNet stats.
        Returns: (B, embed_dim) fused embedding.
        """
        x = x.to(self.device)

        projected = []
        for name in self.encoder_names:
            enc = self.encoders[name]
            if self.cfg.freeze_backbones:
                with torch.no_grad():
                    feat = enc(x)
            else:
                feat = enc(x)
            projected.append(self.projections[name](feat))   # (B, embed_dim)

        # Weighted sum across encoders.
        stacked = torch.stack(projected, dim=0)              # (K, B, embed_dim)
        weights = self.fusion_weights.view(-1, 1, 1)         # (K, 1, 1)
        fused = (weights * stacked).sum(dim=0)               # (B, embed_dim)
        return fused

    # ----- convenience: extract from raw inputs -----

    def _to_tensor(self, image) -> torch.Tensor:
        """Accept filepath / PIL.Image / ndarray / tensor -> (3, H, W) tensor."""
        if isinstance(image, torch.Tensor):
            t = image
            if t.dim() == 4:
                t = t.squeeze(0)
            if t.dim() != 3:
                raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(t.shape)}")
            # If it's not already normalised / resized, do it via PIL roundtrip.
            arr = t.detach().cpu().numpy()
            if arr.max() <= 1.5:
                arr = (arr * 255).clip(0, 255)
            arr = arr.astype(np.uint8)
            if arr.shape[0] == 3:        # (C, H, W) -> (H, W, C)
                arr = arr.transpose(1, 2, 0)
            image = Image.fromarray(arr)
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.dtype != np.uint8:
                if arr.max() <= 1.5:
                    arr = (arr * 255).clip(0, 255)
                arr = arr.astype(np.uint8)
            image = Image.fromarray(arr)
        elif isinstance(image, (str, bytes)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return self._preprocess(image)

    @torch.no_grad()
    def extract(self, image, as_numpy: bool = False) -> Union[torch.Tensor, np.ndarray]:
        """Single image -> (embed_dim,) embedding."""
        was_training = self.training
        self.eval()
        try:
            x = self._to_tensor(image).unsqueeze(0)          # (1, 3, H, W)
            out = self.forward(x).squeeze(0)                 # (embed_dim,)
        finally:
            if was_training:
                self.train()
        return out.cpu().numpy() if as_numpy else out.cpu()

    # Flag the extract pipeline reads to choose the fast (batched) path.
    prefers_batched_extract: bool = True

    @torch.no_grad()
    def extract_batch(
        self, images: Sequence,
        as_numpy: bool = False,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """List of images -> (N, embed_dim) embeddings.

        Processes images in chunks of `batch_size` so a single forward
        through ResNet101 / ViT / DINO sees many images at once. On GPU
        this is the difference between a 30-minute extraction and a
        30-second one; on CPU it still cuts overhead by ~10x because
        framework launch cost amortizes across the batch.

        `batch_size` defaults to 32 -- safe for a single GPU with ~8 GB.
        Lower it if you OOM, raise it if you have headroom.
        """
        was_training = self.training
        self.eval()
        try:
            chunks: list[torch.Tensor] = []
            ranges = range(0, len(images), batch_size)
            if show_progress:
                from tqdm import tqdm
                ranges = tqdm(
                    ranges,
                    total=(len(images) + batch_size - 1) // batch_size,
                    desc="multi_encoder", unit="batch", dynamic_ncols=True,
                )
            for i in ranges:
                chunk = images[i:i + batch_size]
                tensors = [self._to_tensor(im) for im in chunk]
                batch = torch.stack(tensors, dim=0)          # (B, 3, H, W)
                out = self.forward(batch)                    # (B, embed_dim)
                chunks.append(out.cpu())
            result = torch.cat(chunks, dim=0)
        finally:
            if was_training:
                self.train()
        return result.numpy() if as_numpy else result
