"""
Stage-1 feature extractor for AR-image attribution.

Goal: cheap, fixed-length feature vector per image that captures
tokenizer-family fingerprints (FFT peaks at the patch grid, noise-residual
statistics, color-palette stats). 

The output feeds a linear/SVM/small-MLP
4-way classifier {Real, LlamaGen, VAR/HMAR, RAR}.

Design choices:
- Pure numpy + scipy + PIL so it has no heavy deps and is easy to inspect.
- Features are organised as (named) blocks; you can ablate by passing flags.
- Image size is normalised to 256x256 inside the extractor so spectral peaks
  land at predictable bins regardless of the input resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image
from scipy import ndimage


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureConfig:
    image_size: int = 256                # input is resized to (size, size)
    # Tokenizer grid sizes we expect to see peaks at. 16 = LlamaGen / RAR
    # patch size; the others are VAR/HMAR multi-scale grid sizes.
    grid_sizes: Sequence[int] = (8, 10, 13, 16)
    # Width (in FFT bins) of the band we average around each expected peak.
    peak_bin_radius: int = 1
    # Number of radial bins for the radial spectrum profile.
    n_radial_bins: int = 32
    # High-pass filter sigma for the noise-residual block.
    residual_sigma: float = 1.0
    # Whether each block contributes to the output vector.

    use_spectral_peaks: bool = True
    use_radial_profile: bool = True
    use_residual_stats: bool = True
    use_color_stats: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    """ITU-R 601 luminance. Input float32 in [0, 1], shape (H, W, 3)."""
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def _load_and_normalise(path_or_array, size: int) -> np.ndarray:
    """Return float32 RGB image of shape (size, size, 3) in [0, 1]."""
    if isinstance(path_or_array, np.ndarray):
        arr = path_or_array
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    else:
        img = Image.open(path_or_array).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------

def _log_fft_magnitude(gray: np.ndarray) -> np.ndarray:
    """Centred log-magnitude spectrum of a 2D grayscale image."""
    # Hann window kills edge-related leakage that would otherwise dominate
    # the spectrum and drown out the patch-grid peaks.
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fftshift(np.fft.fft2(gray * window))
    mag = np.log1p(np.abs(f))
    return mag


def spectral_peak_features(
    gray: np.ndarray,
    grid_sizes: Sequence[int],
    radius: int,
) -> np.ndarray:
    """
    For each expected tokenizer grid size G, the patch boundaries appear in
    the FFT as energy spikes at frequencies k/G cycles/pixel for k = 1..G/2.
    We average log-magnitude in a small window around each predicted peak,
    relative to a local baseline, giving a "peakiness" score per grid.

    Returns: vector of length len(grid_sizes) * 3
        [peak_strength, baseline, peak_minus_baseline] per grid size.
    """
    h, w = gray.shape
    assert h == w, "Square image expected"
    N = h
    mag = _log_fft_magnitude(gray)
    cy, cx = N // 2, N // 2

    feats = []
    for G in grid_sizes:
        # Frequencies of interest: k * (N / G) bins from the centre,
        # for k = 1, 2, ..., floor(G / 2). Sample along the +x and +y axes
        # (horizontal and vertical patch boundaries).
        peak_vals, base_vals = [], []
        for k in range(1, G // 2 + 1):
            offset = int(round(k * N / G))
            if offset >= N // 2:
                break
            for (dy, dx) in [(0, offset), (offset, 0)]:
                py, px = cy + dy, cx + dx
                # Peak: small window around expected location
                y0, y1 = max(0, py - radius), min(N, py + radius + 1)
                x0, x1 = max(0, px - radius), min(N, px + radius + 1)
                peak_vals.append(mag[y0:y1, x0:x1].mean())
                # Baseline: annulus a few bins away (avoids the peak itself)
                ring_r = radius + 3
                y0b, y1b = max(0, py - ring_r), min(N, py + ring_r + 1)
                x0b, x1b = max(0, px - ring_r), min(N, px + ring_r + 1)
                ring = mag[y0b:y1b, x0b:x1b].copy()
                # Mask out the inner peak window before averaging
                inner_y0 = max(0, (py - radius) - y0b)
                inner_y1 = inner_y0 + (y1 - y0)
                inner_x0 = max(0, (px - radius) - x0b)
                inner_x1 = inner_x0 + (x1 - x0)
                ring[inner_y0:inner_y1, inner_x0:inner_x1] = np.nan
                base_vals.append(np.nanmean(ring))

        peak = float(np.mean(peak_vals)) if peak_vals else 0.0
        base = float(np.nanmean(base_vals)) if base_vals else 0.0
        feats.extend([peak, base, peak - base])

    return np.asarray(feats, dtype=np.float32)


def radial_spectrum_profile(gray: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Azimuthally averaged log-magnitude as a function of radial frequency.
    Normalised to unit sum so it captures the *shape* of the spectrum (e.g.
    1/f^alpha falloff for natural images vs flatter / peaked falloff for AR
    outputs) rather than overall image energy.
    """
    mag = _log_fft_magnitude(gray)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = min(cy, cx)
    bins = np.linspace(0, r_max, n_bins + 1)
    profile = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if mask.any():
            profile[i] = mag[mask].mean()
    s = profile.sum()
    if s > 0:
        profile = profile / s
    return profile


def residual_statistics(gray: np.ndarray, sigma: float) -> np.ndarray:
    """
    High-pass residual via image - Gaussian(image), then compute a handful
    of moment / energy statistics. Real photos have sensor noise + JPEG
    artefacts; AR samples have decoder-imprinted residuals with very
    different distributions.

    Returns 6 features: mean, std, skew, kurt, energy, |grad|-mean.
    """
    blur = ndimage.gaussian_filter(gray, sigma=sigma)
    res = gray - blur

    mean = res.mean()
    std = res.std() + 1e-8
    centred = res - mean
    skew = (centred ** 3).mean() / (std ** 3)
    kurt = (centred ** 4).mean() / (std ** 4) - 3.0
    energy = (res ** 2).mean()

    gx = np.diff(res, axis=1)
    gy = np.diff(res, axis=0)
    grad_mean = (np.abs(gx).mean() + np.abs(gy).mean()) / 2.0

    return np.asarray(
        [mean, std, skew, kurt, energy, grad_mean], dtype=np.float32
    )


def color_statistics(rgb: np.ndarray) -> np.ndarray:
    """
    VQ tokenizers operate on a finite codebook, so generated images tend to
    have constrained colour distributions. We capture this with simple
    palette-style statistics.

    Returns 7 features.
    """
    flat = rgb.reshape(-1, 3)

    # Per-channel mean and std
    ch_mean = flat.mean(axis=0)            # 3
    ch_std = flat.std(axis=0)              # 3

    # Effective palette size: number of unique 5-bit-per-channel buckets,
    # normalised by the maximum possible for a 256x256 image.
    quantised = (flat * 31).astype(np.int32)
    keys = (quantised[:, 0] << 10) | (quantised[:, 1] << 5) | quantised[:, 2]
    unique_ratio = np.unique(keys).size / float(keys.size)

    return np.concatenate(
        [ch_mean, ch_std, [unique_ratio]]
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Stage1FeatureExtractor:
    """Computes a fixed-length feature vector per image."""

    def __init__(self, config: FeatureConfig | None = None):
        self.cfg = config or FeatureConfig()
        self._names: list[str] = []
        self._populate_names()

    def _populate_names(self) -> None:
        c = self.cfg
        if c.use_spectral_peaks:
            for G in c.grid_sizes:
                for tag in ("peak", "base", "peak_minus_base"):
                    self._names.append(f"spec_g{G}_{tag}")
        if c.use_radial_profile:
            for i in range(c.n_radial_bins):
                self._names.append(f"radial_{i:02d}")
        if c.use_residual_stats:
            for tag in ("mean", "std", "skew", "kurt", "energy", "grad"):
                self._names.append(f"res_{tag}")
        if c.use_color_stats:
            for tag in ("r_mean", "g_mean", "b_mean",
                        "r_std", "g_std", "b_std", "palette_ratio"):
                self._names.append(f"color_{tag}")

    @property
    def feature_names(self) -> list[str]:
        return list(self._names)

    @property
    def n_features(self) -> int:
        return len(self._names)

    def extract(self, image) -> np.ndarray:
        """
        image: filepath, PIL.Image, or HxWx3 ndarray (uint8 or float).
        Returns: 1-D float32 vector of length self.n_features.
        """
        c = self.cfg
        rgb = _load_and_normalise(image, c.image_size) #fixed size for processeing needed for the FFT
        gray = _to_gray(rgb) # Turning to gray for FFT to be faster and better! 

        blocks: list[np.ndarray] = []
        if c.use_spectral_peaks:
            blocks.append(
                spectral_peak_features(gray, c.grid_sizes, c.peak_bin_radius)
            )
        if c.use_radial_profile:
            blocks.append(radial_spectrum_profile(gray, c.n_radial_bins))
        if c.use_residual_stats:
            blocks.append(residual_statistics(gray, c.residual_sigma))
        if c.use_color_stats:
            blocks.append(color_statistics(rgb))

        return np.concatenate(blocks).astype(np.float32)

    def extract_batch(self, images) -> np.ndarray:
        """Returns (N, n_features) array."""
        return np.stack([self.extract(im) for im in images], axis=0)