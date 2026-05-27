# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `forensic`, `lens_features`, `multi_encoder`, `spectral_forensic`,
  `spectral_forensic_lens`, and `combined` extractors. Semantic lenses
  (SL / VSL / SSL) are now implemented via `multi_encoder_extractor.py`.
- `extractor_factory.py` + `extractor_config.yaml` for YAML-driven extractor
  selection.
- `NOTES.md` consolidating WIP notes, the quality log, and experimental results
  that were previously scattered across `README.md` and `src/experiment.md`.

### Changed
- Repository restructured:
  - `src/` extractor modules moved into the `src/extractors/` package.
  - Root-level LaTeX sources moved into `tex/`; compiled outputs into
    `tex/build/` (gitignored).
- `README.md` rewritten: cleaned out clipboard contamination in the data tree,
  removed duplicated content, documented all 7 extractors, and clarified
  `solution.py` vs `deep_fake_classifier_pipeline.py`.
- `WORKFLOW.md` lens-status table updated to reflect that semantic lenses are
  implemented; section 6 now lists all extractors and residual methods.

### Removed
- `intro_meme.tex` / `intro_meme.pdf`, `src/test.py`, `src/__pycache__/`.
- `src/README.md` and `src/experiment.md` (merged into `README.md` and
  `NOTES.md`).

## [v1.1.0] - 2026-05-15

### Added
- Comprehensive `WORKFLOW.md` documentation covering:
  - Detailed explanation of autoregressive image generation process
  - Theoretical background on Causal Fingerprints (Xu et al.)
  - Two-stage classification architecture design
  - Feature extraction approach with all 6 lenses documented
- Architecture diagrams (`CF_Architecture.png`, `Causal_Fingerprint_Architecture.png`)
- ASCII art banner in README for visual project identification
- `.gitignore` for proper version control hygiene

### Changed
- README restructured with clearer setup and usage sections
- Documentation now references CHANGELOG for implementation status
- Expanded lens documentation: 3 signal-level lenses (RGB, DCT, QFT) implemented; 3 semantic lenses (SL, VSL, SSL) planned at this point (since implemented — see Unreleased)

### Documentation
- Added detailed explanations of tokenizer families (HMAR, LlamaGen, VAR, RAR)
- Documented Semantic-Invariant Latent Spaces (SILS) approach
- Added visual diagrams for two-stage classification pipeline

## [v1.0.0_one_layer_extractor] - 2026-05-01

### Added
- Feature extraction pipeline with three spectral lenses:
  - RGB lens: pixel-domain statistics on high-pass residual
  - DCT lens: 2D Discrete Cosine Transform statistics per channel
  - QFT lens: grayscale FFT low-frequency band analysis
- Three classifier options: LogisticRegression, LinearSVM, HistGradientBoosting
- CLI commands:
  - `extract`: Extract and cache features from images
  - `train`: Train classifiers and select best model
  - `predict`: Generate predictions on test set
- Python API: `classify_images(img_paths)` for direct classification
- 4-class classification: Real, LlamaGen, VAR/HMAR, RAR
- uv project setup with dependency management
