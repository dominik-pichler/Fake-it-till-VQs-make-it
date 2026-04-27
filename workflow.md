# Start

What am i discriminating? -> autoregressiv generators -> this means  prediciting step by step! p(x1) -> p(x2/x1) -> .... with a loss working over neg. log. likelihood! 
Okay so the next token prediction is based on a distribution p_theta. The sampling happens based on the policy and can be determinstic or stochastic depending on the config. 


Models that I have :
- HMAR -> Multi-scale VQ Tokenization with next-scale prediction
- LlamaGen -> Standard 2D VQ-VAE with raster-order prediction
- VAR -> Multi-scale VQ Tokenization with next-scale prediction
- RAR -> Randomized-order masked AR with a BPE-like tokenizer 


The idea is to classify based on 
    1.  the tokenzier family 
        The AR systems in this exercise work the same way:
        (a) tokenize the image into discrete codes, (b) train a transformer to predict those codes autoregressively, (c) at inference time, sample codes and decode them back to pixels. 
        -> hence the way the tokenization works determines the visual fingerprint! 


    2.  The Capactity -> How big the transformer is


    3. Potential sensor noise (the original images)



## Hence i am build a two strage classifier

Stage 1 (family classifier): spectral features + simple classifier → predicts {Real, LlamaGen-family, VAR/HMAR-family, RAR-family}. Should be very accurate.
Stage 2 (within-family classifier): deep semantic features → splits LlamaGen-B vs LlamaGen-L, VAR-d20 vs VAR-d30, etc. Harder, but tractable.

### High level classifier: 

```
                ┌─────────────┐
                │ Input image │
                └──────┬──────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  STAGE 1: Family classifier  │
        │  (spectral / low-level cues) │
        └──────────────┬───────────────┘
                       │
       ┌───────────┬───┴────┬────────────┐
       ▼           ▼        ▼            ▼
     Real      LlamaGen   VAR/HMAR      RAR
                  │         │            │
                  ▼         ▼            ▼
        ┌──────────────────────────────┐
        │  STAGE 2: Within-family      │
        │  (semantic / deep features)  │
        └──────────────┬───────────────┘
                       │
                       ▼
              One of 9 source labels
```

### Detailed Classifier





```
                          ┌─────────────────────┐
                          │   Input image       │
                          │   (256x256 RGB)     │
                          └──────────┬──────────┘
                                     │
                                     ▼
                  ┌──────────────────────────────────┐
                  │  Feature extraction (spectral)   │
                  │  - 2D FFT magnitude              │
                  │  - Radial / peak features at     │
                  │    1/16, 1/13, 1/10, 1/8 ...     │
                  │  - Noise residual statistics     │
                  └──────────────────┬───────────────┘
                                     │
                                     ▼
              ╔══════════════════════════════════════════╗
              ║  STAGE 1: Family classifier              ║
              ║  (linear / SVM / small MLP, 4-way)       ║
              ╚══════════════════════╤═══════════════════╝
                                     │
        ┌────────────────┬───────────┼───────────────┬────────────────┐
        ▼                ▼           ▼               ▼                ▼
   ┌─────────┐     ┌──────────┐  ┌────────┐     ┌─────────┐    ┌──────────┐
   │  Real   │     │ LlamaGen │  │  VAR / │     │   RAR   │    │ (low     │
   │ImageNet │     │  family  │  │  HMAR  │     │  family │    │ confidence│
   │         │     │          │  │ family │     │         │    │  → fall  │
   │ → DONE  │     │          │  │        │     │         │    │  through)│
   └─────────┘     └─────┬────┘  └────┬───┘     └────┬────┘    └─────┬────┘
                         │            │              │               │
                         ▼            ▼              ▼               ▼
              ╔══════════════════════════════════════════════════════════╗
              ║  STAGE 2: Within-family classifier                       ║
              ║  (ResNet-50 / CLIP features + small head)                ║
              ║  Trained per family OR one shared net w/ family as input ║
              ╚══════════════════════════╤═══════════════════════════════╝
                                         │
        ┌──────────────┬─────────────────┼─────────────┬───────────────┐
        ▼              ▼                 ▼             ▼               ▼
   ┌──────────┐  ┌──────────┐      ┌──────────┐  ┌──────────┐   ┌──────────┐
   │LlamaGen-B│  │LlamaGen-L│      │ VAR-d20  │  │ VAR-d30  │   │  RAR-L   │
   └──────────┘  └──────────┘      │ HMAR-d20 │  │ HMAR-d30 │   │  RAR-XXL │
                                   └──────────┘  └──────────┘   └──────────┘
                                          │             │
                                          │             │     (HMAR vs VAR
                                          ▼             ▼      is the hard
                                    ┌──────────┐  ┌──────────┐  split — needs
                                    │ HMAR-d20 │  │ HMAR-d30 │  finer features
                                    │   vs     │  │   vs     │  or contrastive
                                    │ VAR-d20  │  │ VAR-d30  │  embeddings)
                                    └──────────┘  └──────────┘

Final output: one of 9 source labels
{Real, HMAR-d20, HMAR-d30, LlamaGen-B, LlamaGen-L,
 VAR-d20, VAR-d30, RAR-L, RAR-XXL}
```





