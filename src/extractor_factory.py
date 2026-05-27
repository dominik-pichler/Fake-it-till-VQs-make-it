"""
Factory: build a feature extractor from a YAML config.

The YAML's top-level `extractor:` key picks one of:
  - spectral             -> SpectralFeatureExtractor
  - forensic             -> ForensicFeatureExtractor   (SRM + wavelet + LBP)
  - spectral_forensic    -> SpectralForensicExtractor  (spectral + forensic; pure numpy)
  - lens_features        -> LensFeaturesExtractor      (NLF + CFA + radial-spec + LCA; pure numpy)
  - spectral_forensic_lens -> spectral + forensic + lens_features, concatenated
  - multi_encoder        -> MultiEncoderExtractor
  - combined             -> CombinedFeatureExtractor   (spectral + multi_encoder)

The matching section provides constructor kwargs. See extractor_config.yaml
for the shape of each section.

Config resolution order:
  1. Explicit path passed to build_extractor(config=...)
  2. EXTRACTOR_CONFIG env var
  3. extractor_config.yaml next to this file
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "extractor_config.yaml"


def _resolve_config_path(path: Path | str | None) -> Path:
    if path is not None:
        return Path(path)
    env_path = os.environ.get("EXTRACTOR_CONFIG")
    if env_path:
        return Path(env_path)
    return DEFAULT_CONFIG_PATH


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    with _resolve_config_path(path).open() as f:
        return yaml.safe_load(f) or {}


def build_extractor(config: dict[str, Any] | Path | str | None = None):
    """Return a feature extractor instance based on the YAML config.

    `config` may be:
      - None                      -> default lookup (env var, then default path)
      - str or Path               -> read YAML from that path
      - dict (already-loaded)     -> use as-is
    """
    if config is None or isinstance(config, (str, Path)):
        config = load_config(config)

    kind = config.get("extractor", "spectral")

    if kind == "spectral":
        from extractors.spectral_extractor import FeatureConfig, SpectralFeatureExtractor
        section = config.get("spectral") or {}
        return SpectralFeatureExtractor(FeatureConfig(**section))

    if kind == "forensic":
        from extractors.forensic_extractor import ForensicConfig, ForensicFeatureExtractor
        section = config.get("forensic") or {}
        return ForensicFeatureExtractor(ForensicConfig(**section))

    if kind == "spectral_forensic":
        from extractors.spectral_extractor import FeatureConfig
        from extractors.forensic_extractor import ForensicConfig
        from extractors.spectral_forensic_extractor import SpectralForensicExtractor
        section = config.get("spectral_forensic") or {}
        spec_kwargs = section.get("spectral") or {}
        forn_kwargs = section.get("forensic") or {}
        return SpectralForensicExtractor(
            spectral_config=FeatureConfig(**spec_kwargs),
            forensic_config=ForensicConfig(**forn_kwargs),
        )

    if kind == "lens_features":
        from extractors.lens_features_extractor import (
            LensFeaturesConfig, LensFeaturesExtractor,
        )
        section = config.get("lens_features") or {}
        return LensFeaturesExtractor(LensFeaturesConfig(**section))

    if kind == "spectral_forensic_lens":
        from extractors.spectral_extractor import FeatureConfig
        from extractors.forensic_extractor import ForensicConfig
        from extractors.lens_features_extractor import LensFeaturesConfig
        from extractors.spectral_forensic_lens_extractor import (
            SpectralForensicLensExtractor,
        )
        section = config.get("spectral_forensic_lens") or {}
        spec_kwargs = section.get("spectral") or {}
        forn_kwargs = section.get("forensic") or {}
        lens_kwargs = section.get("lens_features") or {}
        return SpectralForensicLensExtractor(
            spectral_config=FeatureConfig(**spec_kwargs),
            forensic_config=ForensicConfig(**forn_kwargs),
            lens_config=LensFeaturesConfig(**lens_kwargs),
        )

    if kind == "multi_encoder":
        from extractors.multi_encoder_extractor import MultiEncoderConfig, MultiEncoderExtractor
        section = dict(config.get("multi_encoder") or {})
        device = section.pop("device", "cpu")
        return MultiEncoderExtractor(MultiEncoderConfig(**section), device=device)

    if kind == "combined":
        from extractors.spectral_extractor import FeatureConfig
        from extractors.multi_encoder_extractor import MultiEncoderConfig
        from extractors.combined_extractor import CombinedFeatureExtractor
        section = config.get("combined") or {}
        device = section.get("device", "cpu")
        spec_kwargs = section.get("spectral") or {}
        deep_kwargs = section.get("multi_encoder") or {}
        return CombinedFeatureExtractor(
            spectral_config=FeatureConfig(**spec_kwargs),
            deep_config=MultiEncoderConfig(**deep_kwargs),
            device=device,
        )

    raise ValueError(
        f"Unknown extractor type: {kind!r}. Expected one of: "
        f"spectral, forensic, spectral_forensic, lens_features, "
        f"spectral_forensic_lens, multi_encoder, combined."
    )
