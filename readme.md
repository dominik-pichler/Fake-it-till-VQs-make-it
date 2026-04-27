# Can you tell which AI made an image?

Modern autoregressive image generators produce strikingly realistic images, but each model leaves behind subtle fingerprints in its output. In this challenge, you'll build a classifier that can identify which generator produced an image — or whether it's a real image.

## The Challenge

You're given images from 9 sources — 8 different autoregressive generators and real ImageNet photos. Your job is to figure out which generative model made what.

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

Your classifier is evaluated by accuracy on the test data.

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

## What to Do

Edit `solution.py` and implement:

```python
def classify_images(img_paths: List[Path]) -> List[int]:
```

It takes image paths, returns integer labels (0–8). You're free to use any approach — train a model in the container, train externally and upload weights, or try something else entirely.


---

# Current State of Implementation and Next Steps: 




