"""
Combined feature extractor: spectral features + forensic features.

Pure numpy / scipy / PIL -- no torch, no scikit-image, no pywavelets.
Designed to be the default high-quality extractor on CPU-only machines
where MultiEncoderExtractor is too slow.

Output layout (1-D):
    [ spectral features | forensic features ]

The two source extractors use disjoint naming prefixes
(spectral: rgb_/dct_/qft_/..., forensic: srm_/wave_/lbp_) so feature_names
are already unique end-to-end.
"""
from __future__ import annotations

import numpy as np

from .spectral_extractor import FeatureConfig, SpectralFeatureExtractor
from .forensic_extractor import ForensicConfig, ForensicFeatureExtractor


class SpectralForensicExtractor:
    """Wraps SpectralFeatureExtractor and ForensicFeatureExtractor.

    Same public surface as either source extractor:
        - feature_names: list[str]
        - n_features:    int
        - extract(img)        -> (n_features,) np.float32
        - extract_batch(imgs) -> (N, n_features) np.float32
    """

    def __init__(
        self,
        spectral_config: FeatureConfig | None = None,
        forensic_config: ForensicConfig | None = None,
    ):
        self.spectral = SpectralFeatureExtractor(spectral_config)
        self.forensic = ForensicFeatureExtractor(forensic_config)

    @property
    def spectral_dim(self) -> int:
        return self.spectral.n_features

    @property
    def forensic_dim(self) -> int:
        return self.forensic.n_features

    @property
    def n_features(self) -> int:
        return self.spectral_dim + self.forensic_dim

    @property
    def feature_names(self) -> list[str]:
        return self.spectral.feature_names + self.forensic.feature_names

    def extract(self, image) -> np.ndarray:
        spec_vec = self.spectral.extract(image)
        forn_vec = self.forensic.extract(image)
        return np.concatenate([spec_vec, forn_vec]).astype(np.float32)

    def extract_batch(self, images) -> np.ndarray:
        spec = self.spectral.extract_batch(images)
        forn = self.forensic.extract_batch(images)
        return np.concatenate([spec, forn], axis=1).astype(np.float32)
