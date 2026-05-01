# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
