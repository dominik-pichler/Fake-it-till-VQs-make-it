# Can you tell which AI made an image?

Modern autoregressive image generators produce strikingly realistic images, but each model leaves behind subtle fingerprints in its output. In this challenge, you'll build a classifier that can identify which generator produced an image — or whether it's a real image.

## Setup

Images from 9 sources — 8 different autoregressive generators and real ImageNet photos. 
The purpose of this repo is to provie a tool that to figure out which generative model made what.

| Label | Source |
|-------|--------|
| 0 | Real (ImageNet) |
| 1 | HMAR-d20 |
| 2 | HMAR-d30 |
| 3 | LlamaGen-B |
| 4 | LlamaGen-L |
| 5 | VAR-d20 |
| 6 | VAR-d30 |
| 7 | RAR-L |
| 8 | RAR-XXL |


## Data

All images are 256x256 PNGs. The train and test splits use different ImageNet classes.

```
data/
├── train/          # 7,000 images per source (63,000 total)
│   ├── real/
│   ├── hmar_d20/
│   ├── hmar_d30/
│   ├── llamagen_B_VQ-16/
│   ├── llamagen_L_VQ-16/
│   ├── nspvar_20/
│   ├── nspvar_30/
│   ├── rar_l/                      Paste
│   └── rar_xxl/                    Select All
├── val/            # 1,500 images                       al)
│   └── (same structure as train)   How-to disable mouse
└── test/           # 13,500 images, labels are hidden
    ├── 00000.png
    ├── ...
```

For local tests, a reduced set can be extracted via: 

```
ssh -p 2222 <challenge>@<Server URL> 'for dir in /home/user/data/*/*/ /home/user/data/test/; do ls "$dir"*.png 2>/dev/null | head -50; done' | \
```


## How to use
Technical details can be found in the [Manual](src/Readme.md).


It takes image paths, returns integer labels (0–8). You're free to use any approach — train a model in the container, train externally and upload weights, or try something else entirely.


---

# Current State of Implementation and Next Steps: 
Current State of implementation can be found in the [Changelog](CHANGELOG.md)

- [ ] Residuals are currently ony determined by comparsion to gaussian -> needs to be swaped with something else like DIRE (https://arxiv.org/abs/2303.09295)
- [ ] Classifiers are currently only Logistic Regression, Linear SVM , HistGradientBoosting yet, using CNNs seem more promising
  - Could happen via: Encoder: maps 128-dim fingerprint → parameters (μ, σ) of a 64-dim Gaussian






