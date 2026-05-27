# Can you tell which AI made an image?

```
    THE LINEUP
    ══════════════════════════════════════════════════════════════

      ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
      │ 📷  │    │ 🦙  │    │ 🎰  │    │ 🎲  │    │ 🤖  │
      │     │    │     │    │     │    │     │    │     │
      └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘
         │          │          │          │          │
       Real     LlamaGen     VAR        HMAR        RAR

                    🔍
                 ╔══════╗
                 ║ FFT  ║  "I see your frequencies, impostor!"
                 ╚══════╝

    ══════════════════════════════════════════════════════════════
```

Modern autoregressive image generators produce strikingly realistic images, but
each model leaves behind subtle fingerprints in its output. This repo provides a
tool that identifies which generator produced a given image — or whether it's a
real photograph.

## Classes

9 sources: 8 autoregressive generators and real ImageNet photos. All images are
256×256 PNGs; train and test splits use disjoint ImageNet classes.

| Label | Source          |
|-------|-----------------|
| 0     | Real (ImageNet) |
| 1     | HMAR-d20        |
| 2     | HMAR-d30        |
| 3     | LlamaGen-B      |
| 4     | LlamaGen-L      |
| 5     | VAR-d20         |
| 6     | VAR-d30         |
| 7     | RAR-L           |
| 8     | RAR-XXL         |

## Data layout

```
src/data/
├── train/                       # 7,000 images per source (63,000 total)
│   ├── real/
│   ├── hmar_d20/
│   ├── hmar_d30/
│   ├── llamagen_B_VQ-16/
│   ├── llamagen_L_VQ-16/
│   ├── nspvar_20/
│   ├── nspvar_30/
│   ├── rar_l/
│   └── rar_xxl/
├── val/                         # 1,500 images, same structure as train
└── test/                        # 13,500 images, labels hidden
    ├── 00000.png
    └── ...
```

## Repo layout

```
wave_1/
├── README.md         this file
├── WORKFLOW.md       theory, architecture, lens design
├── NOTES.md          experimental results, WIP, open questions
├── CHANGELOG.md      version history
├── tex/              LaTeX figures (build artifacts in tex/build/)
├── imgs/             architecture diagrams referenced by WORKFLOW.md
├── papers/           reference papers
└── src/
    ├── deep_fake_classifier_pipeline.py   train / extract / predict
    ├── solution.py                         single-shot inference entry point
    ├── extractor_factory.py                builds an extractor from YAML
    ├── extractor_config.yaml               which extractor + its kwargs
    ├── extractors/                         feature extractor modules
    ├── data/  features/  models/           gitignored, generated
    ├── pyproject.toml  uv.lock
    └── .venv/
```

## Setup

```shell
cd src
uv sync
```

## How to run

### Inference on a folder of images

Use `solution.py` when you just want labels out of an existing trained model:

```shell
uv run solution.py </path/to/pngs> --model ./<my_model.joblib>
```

For the bundled pipeline:

```shell
uv run solution.py data/test/ --model models/best.joblib
```

It takes image paths and returns integer labels (0–8).

### Full training pipeline

`deep_fake_classifier_pipeline.py` handles feature extraction, training, and
prediction in three subcommands:

```shell
# 1. Extract and cache features
uv run python deep_fake_classifier_pipeline.py extract --data-root data --out features/

# 2. Train classifiers (selects the best by CV)
uv run python deep_fake_classifier_pipeline.py train --features features/ --out models/

# 3. Generate predictions
uv run python deep_fake_classifier_pipeline.py predict \
    --features features/ --model models/best.joblib --out results.csv
```

### Python API

```python
from pathlib import Path
from deep_fake_classifier_pipeline import classify_images, CLASS_NAMES

paths = [Path("image1.png"), Path("image2.png")]
labels = classify_images(paths)

for path, label in zip(paths, labels):
    print(f"{path.name}: {CLASS_NAMES[label]}")
```

## Configuring the feature extractor

Which extractor runs is controlled by the top-level `extractor:` key in
`src/extractor_config.yaml`. Available choices:

| Key                       | What it does                                                                 |
|---------------------------|------------------------------------------------------------------------------|
| `spectral`                | RGB / DCT / QFT lenses on a high-pass residual (pure numpy)                  |
| `forensic`                | SRM + Haar wavelet sub-band + LBP residual moments                           |
| `lens_features`           | NLF, CFA periodicity, radial + angular residual spectrum, LCA               |
| `multi_encoder`           | Frozen ResNet101 / ViT-B/16 / DINO-ResNet50 embeddings (semantic lenses)     |
| `spectral_forensic`       | `spectral` ⊕ `forensic` concatenated                                          |
| `spectral_forensic_lens`  | `spectral` ⊕ `forensic` ⊕ `lens_features` concatenated                        |
| `combined`                | `spectral` ⊕ `multi_encoder` (signal + semantic)                              |

See `src/extractor_config.yaml` for the kwargs each section accepts.

### Residual methods

The lens extractors first compute a residual `r = x − denoise(x)` to suppress
image content. Four denoisers are available (`residual_method:`):

- `gaussian` — `x − Gaussian(x)`. Cheap, leaky.
- `median` — `x − Median(x)`. Edge-preserving.
- `multi_gaussian` — average of Gaussian residuals at multiple σ.
- `wavelet` — Haar wavelet shrinkage. Closest in spirit to classical PRNU work.

## Deploying to the remote SSH host

Copy a single file:

```shell
scp -P 2222 /path/to/local/file <challenge>@<server>:/path/on/server/
```

Sync the project but skip heavy generated directories:

```shell
rsync -av -e "ssh -p 2222" \
  --exclude="features/" \
  --exclude="data/" \
  --exclude=".venv/" \
  /path/to/local/test/project/src/ <challenge>@<server>:/path/on/server/
```

Pull a reduced local sample from the remote (50 images per source):

```shell
ssh -p 2222 <challenge>@<server> \
  'for dir in /home/user/data/*/*/ /home/user/data/test/; do
     ls "$dir"*.png 2>/dev/null | head -50
   done'
```

## Further reading

- [WORKFLOW.md](WORKFLOW.md) — theory: AR generation, causal fingerprints, the
  6-lens design, two-stage classifier architecture.
- [NOTES.md](NOTES.md) — experimental results, quality log, open questions.
- [CHANGELOG.md](CHANGELOG.md) — version history.
- `papers/` — reference papers (Causal Fingerprints, etc.).
