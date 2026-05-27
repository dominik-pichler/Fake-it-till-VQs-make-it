"""
Combined feature extractor: spectral + forensic + lens_features + fingerprint.

Pure-numpy stack that adds the per-class residual fingerprint block on top
of all hand-crafted features. The fingerprint block specifically targets
within-family disambiguation (HMAR_d20<->HMAR_d30 etc.) that hand-crafted
features alone cannot resolve.

Output layout (1-D):
    [ spectral | forensic | lens_features | fingerprint ]

Naming prefixes are disjoint across all four sources (spectral: rgb_/dct_/
qft_/..., forensic: srm_/wave_/lbp_, lens: nlf_/cfa_/resradspec_/...,
fingerprint: fp_ncc_*) so feature_names are globally unique.

Requires that the fingerprint .npz has been pre-computed via:
    python deep_fake_classifier_pipeline.py fingerprints --data-root <data> --out fingerprints.npz
"""
from __future__ import annotations

import numpy as np

from .spectral_extractor import FeatureConfig, SpectralFeatureExtractor
from .forensic_extractor import ForensicConfig, ForensicFeatureExtractor
from .lens_features_extractor import LensFeaturesConfig, LensFeaturesExtractor
from .fingerprint_extractor import FingerprintConfig, FingerprintExtractor


class SpectralForensicLensFingerprintExtractor:
    """spectral + forensic + lens_features + fingerprint, concatenated."""

    def __init__(
        self,
        spectral_config: FeatureConfig | None = None,
        forensic_config: ForensicConfig | None = None,
        lens_config: LensFeaturesConfig | None = None,
        fingerprint_config: FingerprintConfig | None = None,
    ):
        self.spectral = SpectralFeatureExtractor(spectral_config)
        self.forensic = ForensicFeatureExtractor(forensic_config)
        self.lens = LensFeaturesExtractor(lens_config)
        self.fingerprint = FingerprintExtractor(fingerprint_config)

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
    def fingerprint_dim(self) -> int:
        return self.fingerprint.n_features

    @property
    def n_features(self) -> int:
        return (self.spectral_dim + self.forensic_dim
                + self.lens_dim + self.fingerprint_dim)

    @property
    def feature_names(self) -> list[str]:
        return (
            self.spectral.feature_names
            + self.forensic.feature_names
            + self.lens.feature_names
            + self.fingerprint.feature_names
        )

    def extract(self, image) -> np.ndarray:
        spec_vec = self.spectral.extract(image)
        forn_vec = self.forensic.extract(image)
        lens_vec = self.lens.extract(image)
        fp_vec = self.fingerprint.extract(image)
        return np.concatenate(
            [spec_vec, forn_vec, lens_vec, fp_vec]
        ).astype(np.float32)

    def extract_batch(self, images) -> np.ndarray:
        spec = self.spectral.extract_batch(images)
        forn = self.forensic.extract_batch(images)
        lens = self.lens.extract_batch(images)
        fp = self.fingerprint.extract_batch(images)
        return np.concatenate(
            [spec, forn, lens, fp], axis=1
        ).astype(np.float32)
