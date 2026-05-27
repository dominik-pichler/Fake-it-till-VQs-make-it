"""
Combined feature extractor: spectral features + multi-encoder deep embedding.

Convenience wrapper that runs both SpectralFeatureExtractor and
MultiEncoderExtractor and returns their concatenated feature vector.

Output layout (1-D):
    [ spectral features | merged deep embedding ]

Use feature_names / spectral_dim / deep_dim to slice back into blocks
downstream if you want to standardise each block independently.
"""
from __future__ import annotations

import numpy as np
import torch

from spectral_extractor import FeatureConfig, SpectralFeatureExtractor
from multi_encoder_extractor import MultiEncoderConfig, MultiEncoderExtractor


class CombinedFeatureExtractor:
    # The extract pipeline checks this flag to dispatch through extract_batch
    # (which chunks the deep half) instead of looping extract() per image.
    prefers_batched_extract: bool = True

    def __init__(
        self,
        spectral_config: FeatureConfig | None = None,
        deep_config: MultiEncoderConfig | None = None,
        device: str | torch.device = "cpu",
    ):
        self.spectral = SpectralFeatureExtractor(spectral_config)
        self.deep = MultiEncoderExtractor(deep_config, device=device)

    @property
    def spectral_dim(self) -> int:
        return self.spectral.n_features

    @property
    def deep_dim(self) -> int:
        return self.deep.n_features

    @property
    def n_features(self) -> int:
        return self.spectral_dim + self.deep_dim

    @property
    def feature_names(self) -> list[str]:
        deep_names = [f"deep_{i:04d}" for i in range(self.deep_dim)]
        return self.spectral.feature_names + deep_names

    def extract(self, image) -> np.ndarray:
        spec_vec = self.spectral.extract(image)                  # numpy
        deep_vec = self.deep.extract(image, as_numpy=True)       # numpy
        return np.concatenate([spec_vec, deep_vec]).astype(np.float32)

    def extract_batch(self, images,
                      batch_size: int = 32,
                      show_progress: bool = False) -> np.ndarray:
        """Spectral (per-image, cheap) + multi-encoder (chunked GPU pass)."""
        spec = self.spectral.extract_batch(images)               # (N, D_spec)
        deep = self.deep.extract_batch(
            images, as_numpy=True,
            batch_size=batch_size, show_progress=show_progress,
        )                                                         # (N, D_deep)
        return np.concatenate([spec, deep], axis=1).astype(np.float32)
