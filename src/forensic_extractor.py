"""
Forensic feature extractor: SRM + Wavelet + LBP.

Three training-free, CPU-friendly lenses with strong forensic priors:

    1. SRM   -- A curated subset of Steganalysis Rich Model high-pass kernels,
                applied per channel. Generators (VAE/VQ/AR decoders) leave
                low-level noise fingerprints that survive these filters far
                more cleanly than they show up in pixel statistics.
    2. WAVE  -- 2D Haar wavelet decomposition on the grayscale image.
                Multi-resolution view that complements your DCT/FFT lenses
                by localising artefacts in scale. Implemented from scratch
                with numpy (no pywavelets dependency).
    3. LBP   -- Rotation-invariant LBP-style histogram at multiple radii.
                We use a "bit-count" variant (count of neighbours brighter
                than centre) which gives a 9-bin histogram per radius and
                avoids the need for scikit-image.

Pure numpy / scipy / PIL -- no torch, no scikit-image, no pywavelets.

Public surface mirrors SpectralFeatureExtractor so it slots straight into
extractor_factory and the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
from PIL import Image
from scipy.ndimage import convolve as _ndi_convolve


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForensicConfig:
    image_size: int = 256

    # ---- SRM -------------------------------------------------------------
    use_srm: bool = True
    # Residuals are truncated to [-T, +T] after dividing by the per-kernel
    # quantization step Q (see srm_features). Standard SRM uses T=2.
    srm_truncate: float = 2.0
    # Quantization step Q. The classical SRM convention is Q = kernel L1 norm
    # (so the unnormalized residual / Q lives on roughly the same integer
    # scale across kernels). We round to int after division, matching the
    # paper, before clipping at +-T.
    srm_quantize: bool = True

    # ---- Wavelet ---------------------------------------------------------
    use_wavelet: bool = True
    wavelet_levels: int = 3              # Haar-only (numpy native)

    # ---- LBP -------------------------------------------------------------
    use_lbp: bool = True
    lbp_radii: Sequence[int] = (1, 2)
    # Bit-count LBP: at each pixel count how many of the 8 grid neighbours
    # at radius r are brighter than the centre -> integer in [0, 8] ->
    # 9-bin normalised histogram per radius.


# ---------------------------------------------------------------------------
# SRM kernels
# ---------------------------------------------------------------------------
# Curated subset of well-known SRM linear high-pass filters covering 1st-,
# 2nd-, and 3rd-order residuals at multiple orientations. Each kernel is
# stored unnormalised; we normalise to unit L1 at use time.

_SRM_KERNELS: dict[str, np.ndarray] = {
    # 1st-order
    "first_h": np.array([[0,  0, 0],
                         [-1, 1, 0],
                         [0,  0, 0]], dtype=np.float32),
    "first_v": np.array([[0, -1, 0],
                         [0,  1, 0],
                         [0,  0, 0]], dtype=np.float32),
    # 2nd-order
    "second_h": np.array([[0,  0, 0],
                          [1, -2, 1],
                          [0,  0, 0]], dtype=np.float32),
    "second_v": np.array([[0,  1, 0],
                          [0, -2, 0],
                          [0,  1, 0]], dtype=np.float32),
    "laplacian": np.array([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]], dtype=np.float32),
    # Edge3 (square ring, often called KB-edge)
    "edge3": np.array([[-1,  2, -1],
                       [ 2, -4,  2],
                       [-1,  2, -1]], dtype=np.float32),
    # 3rd-order (5x5)
    "third_h": np.array([[0, 0,  0, 0, 0],
                         [0, 0,  0, 0, 0],
                         [1,-3,  3,-1, 0],
                         [0, 0,  0, 0, 0],
                         [0, 0,  0, 0, 0]], dtype=np.float32),
    "third_v": np.array([[0, 0,  1, 0, 0],
                         [0, 0, -3, 0, 0],
                         [0, 0,  3, 0, 0],
                         [0, 0, -1, 0, 0],
                         [0, 0,  0, 0, 0]], dtype=np.float32),
    # Square5 (5x5 2nd-order centred)
    "square5": np.array([[-1,  2, -2,  2, -1],
                         [ 2, -6,  8, -6,  2],
                         [-2,  8,-12,  8, -2],
                         [ 2, -6,  8, -6,  2],
                         [-1,  2, -2,  2, -1]], dtype=np.float32),
}


def _l1_normalised(k: np.ndarray) -> np.ndarray:
    """Scale a kernel so |k|.sum() == 1 (keeps residuals comparable across filters)."""
    s = np.abs(k).sum()
    return k if s == 0 else (k / s).astype(np.float32)


_SRM_KERNELS_NORM: dict[str, np.ndarray] = {
    name: _l1_normalised(k) for name, k in _SRM_KERNELS.items()
}

# Per-kernel quantization step Q = L1 norm of the unnormalized kernel. The
# classical SRM residual is (unnormalized_kernel * channel) / Q, then rounded
# and clipped to +-T. We precompute Q here so srm_features can convolve with
# the unnormalized kernel and divide once.
_SRM_KERNEL_Q: dict[str, float] = {
    name: float(np.abs(k).sum()) for name, k in _SRM_KERNELS.items()
}


# ---------------------------------------------------------------------------
# Shared helpers (kept compatible with the spectral extractor's conventions)
# ---------------------------------------------------------------------------

def _load_and_normalise(path_or_array, size: int) -> np.ndarray:
    """Return float32 RGB image of shape (size, size, 3) in [0, 1]."""
    if isinstance(path_or_array, np.ndarray):
        arr = path_or_array
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    else:
        img = Image.open(path_or_array).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _moment_stats(x: np.ndarray) -> np.ndarray:
    """5 moments: mean, std, skew, kurt, energy."""
    flat = x.ravel().astype(np.float32)
    mean = flat.mean()
    std = flat.std() + 1e-8
    centred = flat - mean
    skew = (centred ** 3).mean() / (std ** 3)
    kurt = (centred ** 4).mean() / (std ** 4) - 3.0
    energy = (flat ** 2).mean()
    return np.asarray([mean, std, skew, kurt, energy], dtype=np.float32)


_MOMENT_TAGS = ("mean", "std", "skew", "kurt", "energy")


# ---------------------------------------------------------------------------
# Lens 1: SRM
# ---------------------------------------------------------------------------

def srm_features(rgb: np.ndarray, truncate: float,
                 quantize: bool = True) -> np.ndarray:
    """Per-channel SRM residuals; truncated; reduced to moment stats.

    Classical SRM definition (Fridrich & Kodovsky 2012):
        r = round( (unnormalized_kernel * channel) / Q )
        r = clip(r, -T, +T)
    where Q is the L1 norm of the unnormalized kernel and T = truncate.

    The previous version L1-normalized the kernel BEFORE convolution and
    clipped at +-2; since channel in [0, 1] and ||k||_1 = 1, that residual
    was bounded by 1, so the clip never triggered and the truncation step
    was a no-op. We now convolve with the unnormalized kernel, divide by Q,
    optionally round, then clip -- so truncate=2 actually removes outliers.

    Returns 3 channels x N filters x 5 stats features.
    """
    feats = []
    for c in range(3):
        channel = rgb[..., c].astype(np.float32, copy=False)
        # Scale to 0..255 so the integer residual scale matches the SRM paper.
        channel_q = channel * 255.0
        for name, kernel_unnorm in _SRM_KERNELS.items():
            Q = _SRM_KERNEL_Q[name]
            residual = _ndi_convolve(channel_q, kernel_unnorm, mode="reflect") / Q
            if quantize:
                np.rint(residual, out=residual)
            np.clip(residual, -truncate, truncate, out=residual)
            feats.append(_moment_stats(residual))
    return np.concatenate(feats).astype(np.float32)


# ---------------------------------------------------------------------------
# Lens 2: Haar wavelet (numpy-only)
# ---------------------------------------------------------------------------

def _haar_dwt2_step(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One level of 2D Haar DWT.

    Returns (cA, cH, cV, cD) where:
        cA -- approximation (low-pass both axes)
        cH -- horizontal detail (low rows, high cols)
        cV -- vertical detail   (high rows, low cols)
        cD -- diagonal detail   (high both)

    Input is trimmed to even side lengths.
    """
    h, w = x.shape
    h -= h % 2
    w -= w % 2
    x = x[:h, :w]

    # 2x2 blocks: a b
    #             c d
    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]

    cA = (a + b + c + d) * 0.5
    cH = (a + b - c - d) * 0.5
    cV = (a - b + c - d) * 0.5
    cD = (a - b - c + d) * 0.5
    return cA, cH, cV, cD


def wavelet_features(gray: np.ndarray, levels: int) -> np.ndarray:
    """Multi-level Haar 2D-DWT; moment stats per sub-band.

    Total bands = 3*levels + 1 (3 details per level + final approximation).
    Output order matches the pywt convention: [cA_L, then per level deepest-
    first: cH, cV, cD].
    """
    cA = gray.astype(np.float32, copy=True)
    detail_stack = []  # list of (cH, cV, cD), shallowest-first
    for _ in range(levels):
        cA, cH, cV, cD = _haar_dwt2_step(cA)
        detail_stack.append((cH, cV, cD))

    feats = [_moment_stats(cA)]
    # Emit deepest level first so naming matches feature_names ordering
    for cH, cV, cD in reversed(detail_stack):
        feats.append(_moment_stats(cH))
        feats.append(_moment_stats(cV))
        feats.append(_moment_stats(cD))
    return np.concatenate(feats).astype(np.float32)


# ---------------------------------------------------------------------------
# Lens 3: Bit-count LBP (numpy-only)
# ---------------------------------------------------------------------------

# 8 grid neighbour offsets in counter-clockwise order starting from East.
# These are the relative (dy, dx) at unit radius; we scale by r at runtime.
_LBP_NEIGHBOUR_DIRS = [
    ( 0,  1),  # E
    (-1,  1),  # NE
    (-1,  0),  # N
    (-1, -1),  # NW
    ( 0, -1),  # W
    ( 1, -1),  # SW
    ( 1,  0),  # S
    ( 1,  1),  # SE
]


def lbp_features(gray: np.ndarray, radii: Sequence[int]) -> np.ndarray:
    """Bit-count LBP histograms per radius.

    At each pixel, count how many of the 8 grid neighbours at radius r are
    >= the centre. That count is in [0, 8] -> 9-bin normalised histogram.
    """
    gray_i = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.int32)
    feats = []
    for r in radii:
        padded = np.pad(gray_i, r, mode="edge")
        h, w = gray_i.shape
        centre = padded[r:r + h, r:r + w]
        count = np.zeros((h, w), dtype=np.int32)
        for dy, dx in _LBP_NEIGHBOUR_DIRS:
            ny = r + dy * r
            nx = r + dx * r
            nbr = padded[ny:ny + h, nx:nx + w]
            count += (nbr >= centre).astype(np.int32)
        hist, _ = np.histogram(count.ravel(), bins=9, range=(0, 9), density=False)
        hist = hist.astype(np.float32)
        s = hist.sum()
        if s > 0:
            hist = hist / s
        feats.append(hist)
    return np.concatenate(feats).astype(np.float32)


_LBP_BINS_PER_RADIUS = 9


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ForensicFeatureExtractor:
    """Hand-crafted forensic features (SRM + wavelet + LBP).

    Mirrors the public surface of SpectralFeatureExtractor:
        - feature_names: list[str]
        - n_features:    int
        - extract(img)        -> (n_features,) np.float32
        - extract_batch(imgs) -> (N, n_features) np.float32
    """

    def __init__(self, config: ForensicConfig | None = None):
        self.cfg = config or ForensicConfig()
        self._names: list[str] = []
        self._populate_names()

    def _populate_names(self) -> None:
        c = self.cfg

        if c.use_srm:
            for ch in ("r", "g", "b"):
                for kname in _SRM_KERNELS_NORM:
                    for tag in _MOMENT_TAGS:
                        self._names.append(f"srm_{ch}_{kname}_{tag}")

        if c.use_wavelet:
            # Approximation at the deepest level
            for tag in _MOMENT_TAGS:
                self._names.append(f"wave_cA{c.wavelet_levels}_{tag}")
            # Detail sub-bands per level (deepest first, matching pywt order)
            for level in range(c.wavelet_levels, 0, -1):
                for band in ("H", "V", "D"):
                    for tag in _MOMENT_TAGS:
                        self._names.append(f"wave_c{band}{level}_{tag}")

        if c.use_lbp:
            for r in c.lbp_radii:
                for i in range(_LBP_BINS_PER_RADIUS):
                    self._names.append(f"lbp_r{r}_b{i}")

    @property
    def feature_names(self) -> list[str]:
        return list(self._names)

    @property
    def n_features(self) -> int:
        return len(self._names)

    def extract(self, image) -> np.ndarray:
        c = self.cfg
        rgb = _load_and_normalise(image, c.image_size)
        gray = _to_gray(rgb)

        blocks: list[np.ndarray] = []
        if c.use_srm:
            blocks.append(srm_features(rgb, c.srm_truncate, c.srm_quantize))
        if c.use_wavelet:
            blocks.append(wavelet_features(gray, c.wavelet_levels))
        if c.use_lbp:
            blocks.append(lbp_features(gray, c.lbp_radii))
        return np.concatenate(blocks).astype(np.float32)

    def extract_batch(self, images) -> np.ndarray:
        return np.stack([self.extract(im) for im in images], axis=0)
