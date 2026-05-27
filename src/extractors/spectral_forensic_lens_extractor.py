"""
Combined feature extractor: spectral + forensic + lens_features.

Pure-numpy drop-in replacement for ``spectral_forensic`` when you want the
real-camera-provenance block layered on top of the existing hand-crafted
ones. The lens block adds NLF / CFA / radial-residual-spectrum / LCA
features that explicitly characterise "real photo" by positive capture
traces, attacking the low real-class recall a shrinkage-only residual
pipeline tends to produce.

Output layout (1-D):
    [ spectral features | forensic features | lens features ]

Naming prefixes are disjoint across all three sources (spectral:
rgb_/dct_/qft_/..., forensic: srm_/wave_/lbp_, lens: nlf_/cfa_/resradspec_/
lca_) so feature_names are globally unique.
"""
from __future__ import annotations

import numpy as np

from .spectral_extractor import FeatureConfig, SpectralFeatureExtractor
from .forensic_extractor import ForensicConfig, ForensicFeatureExtractor
from .lens_features_extractor import LensFeaturesConfig, LensFeaturesExtractor


class SpectralForensicLensExtractor:
    """spectral + forensic + lens_features, concatenated."""

    def __init__(
        self,
        spectral_config: FeatureConfig | None = None,
        forensic_config: ForensicConfig | None = None,
        lens_config: LensFeaturesConfig | None = None,
    ):
        self.spectral = SpectralFeatureExtractor(spectral_config)
        self.forensic = ForensicFeatureExtractor(forensic_config)
        self.lens = LensFeaturesExtractor(lens_config)

    @property
    def spectral_dim(self) -> int:
        return self.spectral.n_features

    @property
    def forensic_dim(self) -> int:
        return self.forensic.n_features

    @property
    def lens_dim(self) -> int:
        return self.lens.n_features

    @property
    def n_features(self) -> int:
        return self.spectral_dim + self.forensic_dim + self.lens_dim

    @property
    def feature_names(self) -> list[str]:
        return (
            self.spectral.feature_names
            + self.forensic.feature_names
            + self.lens.feature_names
        )

    def extract(self, image) -> np.ndarray:
        spec_vec = self.spectral.extract(image)
        forn_vec = self.forensic.extract(image)
        lens_vec = self.lens.extract(image)
        return np.concatenate([spec_vec, forn_vec, lens_vec]).astype(np.float32)

    def extract_batch(self, images) -> np.ndarray:
        spec = self.spectral.extract_batch(images)
        forn = self.forensic.extract_batch(images)
        lens = self.lens.extract_batch(images)
        return np.concatenate([spec, forn, lens], axis=1).astype(np.float32)
