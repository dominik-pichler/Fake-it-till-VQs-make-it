# Deepfake Image Classifier

A hobby project exploring how to detect AI-generated images by analyzing their spectral fingerprints.

## Background

I got curious about whether different image generators leave behind detectable "fingerprints" - subtle patterns in the frequency domain that reveal how an image was made. Turns out they do! Autoregressive models like LlamaGen, VAR, and RAR each use different tokenization strategies that create unique artifacts.

This project implements a simple classifier that can distinguish between:
- Real photographs
- LlamaGen outputs
- VAR/HMAR outputs
- RAR outputs

## How It Works

Instead of training a deep neural network, I'm using hand-crafted spectral features:

1. **RGB lens** - Statistics on the high-pass filtered image
2. **DCT lens** - Discrete Cosine Transform to catch periodic artifacts
3. **QFT lens** - Low-frequency FFT patterns

These features feed into a simple classifier (logistic regression, SVM, or gradient boosting). It's surprisingly effective for distinguishing generator families.

## Quick Start

```bash
# Install dependencies
uv sync

# Extract features from your images
uv run python deep_fake_classifier_pipeline.py extract --data-root data --out features/

# Train the classifier
uv run python deep_fake_classifier_pipeline.py train --features features/ --out models/

# Classify new images
uv run python deep_fake_classifier_pipeline.py predict --features features/ --model models/best.joblib --out results.csv
```



## Data Structure

```
data/
├── train/
│   ├── real/
│   ├── llamagen_B_VQ-16/
│   ├── llamagen_L_VQ-16/
│   ├── hmar_d20/
│   ├── hmar_d30/
│   ├── nspvar_20/
│   ├── nspvar_30/
│   ├── rar_l/
│   └── rar_xxl/
├── val/
│   └── (same structure)
└── test/
    └── (flat - just images)
```

## Python API

```python
from pathlib import Path
from deep_fake_classifier_pipeline import classify_images, CLASS_NAMES

paths = [Path("image1.png"), Path("image2.png")]
labels = classify_images(paths)

for path, label in zip(paths, labels):
    print(f"{path.name}: {CLASS_NAMES[label]}")
```

## Future Ideas

- Add semantic features using pretrained models (DINO, CLIP)
- Stage 2 classifier to distinguish within families (e.g., LlamaGen-B vs LlamaGen-L)
- Experiment with different residual extraction methods

## References

This project is inspired by the causal fingerprint approach from [Xu et al.](https://arxiv.org/pdf/2509.15406)
