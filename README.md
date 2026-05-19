# Can you tell which AI made an image?

```
    THE LINEUP
    ══════════════════════════════════════════════════════════════

      ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
      │ 📷  │    │🦙   │    │ 🎰  │    │ 🎲  │    │ 🤖  │
      │     │    │     │    │     │    │     │    │     │
      └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘
         │          │          │          │          │
       Real     LlamaGen    VAR       HMAR        RAR

                    🔍
                 ╔══════╗
                 ║ FFT  ║  "I see your frequencies, impostor!"
                 ╚══════╝

    ══════════════════════════════════════════════════════════════
```

Modern autoregressive image generators produce strikingly realistic images, but each model leaves behind subtle fingerprints in its output. This tool identifies which generator produced an image — or whether it's real.

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

```shell
ssh -p 2222 <challenge>@<Server URL> 'for dir in /home/user/data/*/*/ /home/user/data/test/; do ls "$dir"*.png 2>/dev/null | head -50; done' | \
```



## How to deploy to remote ssh

```shell
scp -P 2222 /path/to/local/file <challenge>@<Server URL>:/path/on/server/

```

If you dont need to copy everything, you can use: 
```shell
rsync -av -e "ssh -p 2222" \
  --exclude="features/" \
  --exclude="data/" \
  --exclude=".venv/" \
  /path/to/local/test/project/src/ <challenge>@<Server URL>:/path/on/server/
```

## How to use

If you are just interested in running the classifier, run: 
`uv run solution.py </path/to/pngs> --model ./<my_model.joblib>`

To run the standard pipeline:  
`uv run solution.py data/test/ --model models/best.joblib`

Technical details can be found in the [Manual](src/README.md).


It takes image paths, returns integer labels (0–8). You're free to use any approach — train a model in the container, train externally and upload weights, or try something else entirely.


---

# Extractors

# Residual Calculations
Four methods are available; all are pure numpy + scipy:
- gaussian:  $input - Gaussian(input)$ -> (cheap, leaky)
- median:    $input - Median(input)$               (edge-preserving)
- multi_gaussian:  $average \ of \ Gaussian  \ residuals \ at \ multiple \ sigmas$
- wavelet:         $Haar \ wavelet \ shrinkage \ denoiser$; closest in spirit  to classical PRNU work

# Next Steps: 
Current State of implementation can be found in the [Changelog](CHANGELOG.md)

- [ ] Residuals are currently ony determined by comparison to gaussian -> needs to be swaped with something else like DIRE (https://arxiv.org/abs/2303.09295)
    - Different residual determination models are evaluated and documented in [the experiment](src/experiment.md)
- [ ] Detecting Real images is apparently still very hard for the Classifiers -> maybe I can find some suitable metrics? 
- [ ] Intra-Family Classifier is still not performing well - this needs to be improved 
- [ ] Classifiers are currently only Logistic Regression, Linear SVM , HistGradientBoosting yet, using CNNs seem more promising
      Could happen via: Encoder: maps 128-dim fingerprint → parameters (μ, σ) of a 64-dim Gaussian
- [ ] ~~Currently only spectral Extractor is used - but does already deliver solid results in the familiy classfication. In the next step the encoder extractor + a combination of both should also be tested and evaluated~~
- [ ] ~~Second Stage Classifier has only model type (Logistic Regression: end-to-end fine val acc (hard routing): 0.3763) -> This should be extended to three just as in step 1~~



# Quality Log
- **First run:** (Spectral Extractor, 3 Models ST1, 1 Model Head ST2) end-to-end fine val acc (hard routing): 0.3763
- **Second run:** (Spectral Extractor, 3 Models ST1, 3 Model Head ST2) end-to-end fine val acc (hard routing): 0.3776 --> so my assumption here is that it's an feature issue not a model issue.
- **Third run:** Tested the encoders but this is not a feasable solution due to the very very slow processing on the container (~ 6.5sec/Image) caused by multiple factors: 

  - Issues with mkldnn -> hence 30x slower
    
  ```python
      import torch
      from torchvision.models import resnet101, ResNet101_Weights
      m = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2).eval()
      x = torch.randn(1, 3, 256, 256)
      with torch.no_grad():
          y = m(x)
      print('ok', y.shape)
    ```
  - CUDA not available: 
    ```shell
    python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
    False 0
    ``` 
        
  