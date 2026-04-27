"""
Stage-1 feature extractor for AR-image attribution.

Reorganised around the three training-free latent spaces from the paper (https://arxiv.org/pdf/2509.15406) 

    1. RGB space  -- pixel-domain statistics
    2. DCT space  -- per-channel 2D Discrete Cosine Transform statistics
    3. QFT space  -- grayscale FFT, low-frequency band only

Each lens is applied to a high-pass residual (image - Gaussian(image)) so that
content is suppressed and tokenizer/decoder artefacts are emphasised. This is
a cheap stand-in for a DIRE-style reconstruction residual; swap _compute_residual
for a real reconstruction model later if desired.

The output feeds a linear/SVM/small-MLP 4-way classifier
{Real, LlamaGen, VAR/HMAR, RAR}.

Design choices:
- Pure numpy + scipy + PIL.
- Features grouped by lens; ablate by toggling the use_* flags.
- Image normalised to 256x256 so spectral peaks land at predictable bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.fft import dct


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureConfig:
    image_size: int = 256                # input is resized to (size, size)

    # Tokenizer grid sizes we expect to see peaks at. 16 = LlamaGen / RAR
    # patch size; the others are VAR/HMAR multi-scale grid sizes.
    grid_sizes: Sequence[int] = (8, 10, 13, 16)

    # Width (in FFT/DCT bins) of the band we average around each expected peak.
    peak_bin_radius: int = 1

    # Number of radial bins for the QFT radial profile.
    n_radial_bins: int = 32

    # QFT low-frequency cutoff: keep only frequencies whose radius (in bins)
    # is <= this fraction of the Nyquist limit. 0.25 keeps the inner quarter.
    qft_low_freq_fraction: float = 0.25

    # High-pass filter sigma for computing the residual.
    residual_sigma: float = 1.0

    # Toggle each lens.
    use_rgb_lens: bool = True
    use_dct_lens: bool = True
    use_qft_lens: bool = True


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


def _moment_stats(x: np.ndarray) -> np.ndarray:
    """Six summary moments of a flat array: mean, std, skew, kurt, energy, |grad|."""
    flat = x.ravel()
    mean = flat.mean()
    std = flat.std() + 1e-8
    centred = flat - mean
    skew = (centred ** 3).mean() / (std ** 3)
    kurt = (centred ** 4).mean() / (std ** 4) - 3.0
    energy = (flat ** 2).mean()

    if x.ndim >= 2:
        gx = np.diff(x, axis=-1)
        gy = np.diff(x, axis=-2)
        grad_mean = (np.abs(gx).mean() + np.abs(gy).mean()) / 2.0
    else:
        grad_mean = np.abs(np.diff(flat)).mean()

    return np.asarray([mean, std, skew, kurt, energy, grad_mean], dtype=np.float32)


def _compute_residual(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """
    High-pass residual: image - Gaussian(image), per channel.
    Stand-in for a DIRE-style reconstruction residual. Same shape as input.
    """
    res = np.empty_like(rgb)
    for c in range(rgb.shape[-1]):
        res[..., c] = rgb[..., c] - ndimage.gaussian_filter(rgb[..., c], sigma=sigma)
    return res


# ---------------------------------------------------------------------------
# Lens 1: RGB space
# ---------------------------------------------------------------------------

def rgb_lens_features(residual_rgb: np.ndarray) -> np.ndarray:
    """
    Pixel-domain statistics on the residual.

    Returns:
      - 6 moment stats on grayscale residual
      - 6 stats per channel x 3 channels = 18
      - 3 channel means + 3 channel stds (overall colour bias of residual)
      - 1 palette ratio on the *original* RGB (unique-colour density at 5 bits/ch)

    Total: 6 + 18 + 6 + 1 = 31 features.

    Notes:
      The palette ratio is computed below in extract() because it needs the
      original (non-residual) RGB image. Here we only handle the residual.
    """
    gray_res = _to_gray(residual_rgb)
    blocks = [_moment_stats(gray_res)]
    for c in range(3):
        blocks.append(_moment_stats(residual_rgb[..., c]))
    flat = residual_rgb.reshape(-1, 3)
    blocks.append(np.concatenate([flat.mean(axis=0), flat.std(axis=0)]).astype(np.float32))
    return np.concatenate(blocks).astype(np.float32)


def palette_ratio(rgb: np.ndarray) -> np.ndarray:
    """
    Effective palette size on the *original* image: fraction of distinct
    5-bit-per-channel colour buckets used. VQ tokenizers tend to produce
    constrained colour distributions, so this should be smaller for AR
    outputs than for real photos.
    """
    flat = rgb.reshape(-1, 3)
    quantised = (flat * 31).astype(np.int32)
    keys = (quantised[:, 0] << 10) | (quantised[:, 1] << 5) | quantised[:, 2]
    unique_ratio = np.unique(keys).size / float(keys.size)
    return np.asarray([unique_ratio], dtype=np.float32)


# ---------------------------------------------------------------------------
# Lens 2: DCT space
# ---------------------------------------------------------------------------

def _dct2(channel: np.ndarray) -> np.ndarray:
    """2D DCT-II with orthonormal normalisation."""
    return dct(dct(channel, axis=0, norm="ortho"), axis=1, norm="ortho")


def dct_lens_features(
    residual_rgb: np.ndarray,
    grid_sizes: Sequence[int],
    radius: int,
    n_radial_bins: int,
) -> np.ndarray:
    """
    Per-channel 2D DCT of the residual. The DC coefficient is at (0, 0)
    and frequency increases toward the bottom-right. Patch-grid artefacts
    show up as energy concentrations at frequencies tied to the grid size.

    For each of the 3 channels we extract:
      - 6 moment stats on log(|DCT| + eps)
      - peakiness at expected grid frequencies (len(grid_sizes) * 3 features)
      - radial profile of |DCT| (n_radial_bins features)

    Returns: 3 * (6 + 3*len(grid_sizes) + n_radial_bins) features.
    """
    feats = []
    for c in range(3):
        coeffs = _dct2(residual_rgb[..., c])
        log_mag = np.log1p(np.abs(coeffs))

        # 1) Moments
        feats.append(_moment_stats(log_mag))

        # 2) Peakiness at grid-related frequencies.
        # In the DCT, frequency k along an axis means k full cycles across
        # the image, so a grid of cell size G should leave energy near
        # k = N/G, 2N/G, ... along that axis.
        N = log_mag.shape[0]
        peak_block = []
        for G in grid_sizes:
            peak_vals, base_vals = [], []
            for kk in range(1, G // 2 + 1):
                offset = int(round(kk * N / G))
                if offset >= N:
                    break
                for (dy, dx) in [(0, offset), (offset, 0)]:
                    y0, y1 = max(0, dy - radius), min(N, dy + radius + 1)
                    x0, x1 = max(0, dx - radius), min(N, dx + radius + 1)
                    peak_vals.append(log_mag[y0:y1, x0:x1].mean())
                    ring_r = radius + 3
                    y0b, y1b = max(0, dy - ring_r), min(N, dy + ring_r + 1)
                    x0b, x1b = max(0, dx - ring_r), min(N, dx + ring_r + 1)
                    ring = log_mag[y0b:y1b, x0b:x1b].copy()
                    iy0 = max(0, (dy - radius) - y0b)
                    iy1 = iy0 + (y1 - y0)
                    ix0 = max(0, (dx - radius) - x0b)
                    ix1 = ix0 + (x1 - x0)
                    ring[iy0:iy1, ix0:ix1] = np.nan
                    base_vals.append(np.nanmean(ring))
            peak = float(np.mean(peak_vals)) if peak_vals else 0.0
            base = float(np.nanmean(base_vals)) if base_vals else 0.0
            peak_block.extend([peak, base, peak - base])
        feats.append(np.asarray(peak_block, dtype=np.float32))

        # 3) Radial profile (DC at corner -> radius from (0, 0))
        h, w = log_mag.shape
        y, x = np.indices((h, w))
        r = np.sqrt(y ** 2 + x ** 2)
        r_max = float(np.sqrt((h - 1) ** 2 + (w - 1) ** 2))
        bins = np.linspace(0, r_max, n_radial_bins + 1)
        profile = np.zeros(n_radial_bins, dtype=np.float32)
        for i in range(n_radial_bins):
            mask = (r >= bins[i]) & (r < bins[i + 1])
            if mask.any():
                profile[i] = log_mag[mask].mean()
        s = profile.sum()
        if s > 0:
            profile = profile / s
        feats.append(profile)

    return np.concatenate(feats).astype(np.float32)


# ---------------------------------------------------------------------------
# Lens 3: QFT space (grayscale FFT, low-frequency band)
# ---------------------------------------------------------------------------

def _log_fft_magnitude(gray: np.ndarray) -> np.ndarray:
    """Centred log-magnitude spectrum with a Hann window to reduce leakage."""
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fftshift(np.fft.fft2(gray * window))
    return np.log1p(np.abs(f))


def qft_lens_features(
    residual_rgb: np.ndarray,
    grid_sizes: Sequence[int],
    radius: int,
    n_radial_bins: int,
    low_freq_fraction: float,
) -> np.ndarray:
    """
    Convert residual to grayscale, take FFT, keep only low-frequency
    components inside a circular mask of radius (low_freq_fraction * Nyquist).

    Extracts:
      - 6 moment stats on the low-freq band
      - Peakiness at expected grid frequencies (len(grid_sizes) * 3 features)
      - Normalised radial profile inside the low-freq band (n_radial_bins)

    Total: 6 + 3*len(grid_sizes) + n_radial_bins features.
    """
    gray_res = _to_gray(residual_rgb)
    mag = _log_fft_magnitude(gray_res)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_nyquist = float(min(cy, cx))
    cutoff = low_freq_fraction * r_nyquist

    low_mask = r <= cutoff
    low_band = mag.copy()
    low_band[~low_mask] = 0.0
    low_values = mag[low_mask]

    feats = [_moment_stats(low_values)]

    # Peakiness at expected grid frequencies (only those inside the band)
    N = h
    peak_block = []
    for G in grid_sizes:
        peak_vals, base_vals = [], []
        for kk in range(1, G // 2 + 1):
            offset = int(round(kk * N / G))
            if offset > cutoff:
                break
            for (dy, dx) in [(0, offset), (offset, 0)]:
                py, px = cy + dy, cx + dx
                y0, y1 = max(0, py - radius), min(N, py + radius + 1)
                x0, x1 = max(0, px - radius), min(N, px + radius + 1)
                peak_vals.append(mag[y0:y1, x0:x1].mean())
                ring_r = radius + 3
                y0b, y1b = max(0, py - ring_r), min(N, py + ring_r + 1)
                x0b, x1b = max(0, px - ring_r), min(N, px + ring_r + 1)
                ring = mag[y0b:y1b, x0b:x1b].copy()
                iy0 = max(0, (py - radius) - y0b)
                iy1 = iy0 + (y1 - y0)
                ix0 = max(0, (px - radius) - x0b)
                ix1 = ix0 + (x1 - x0)
                ring[iy0:iy1, ix0:ix1] = np.nan
                base_vals.append(np.nanmean(ring))
        peak = float(np.mean(peak_vals)) if peak_vals else 0.0
        base = float(np.nanmean(base_vals)) if base_vals else 0.0
        peak_block.extend([peak, base, peak - base])
    feats.append(np.asarray(peak_block, dtype=np.float32))

    # Normalised radial profile inside the low-freq band
    bins = np.linspace(0, cutoff, n_radial_bins + 1)
    profile = np.zeros(n_radial_bins, dtype=np.float32)
    for i in range(n_radial_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if mask.any():
            profile[i] = mag[mask].mean()
    s = profile.sum()
    if s > 0:
        profile = profile / s
    feats.append(profile)

    return np.concatenate(feats).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Stage1FeatureExtractor:
    """Computes a fixed-length feature vector per image across three lenses."""

    def __init__(self, config: FeatureConfig | None = None):
        self.cfg = config or FeatureConfig()
        self._names: list[str] = []
        self._populate_names()

    def _populate_names(self) -> None:
        c = self.cfg
        if c.use_rgb_lens:
            for tag in ("mean", "std", "skew", "kurt", "energy", "grad"):
                self._names.append(f"rgb_gray_{tag}")
            for ch in ("r", "g", "b"):
                for tag in ("mean", "std", "skew", "kurt", "energy", "grad"):
                    self._names.append(f"rgb_{ch}_{tag}")
            for tag in ("r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"):
                self._names.append(f"rgb_resid_{tag}")
            self._names.append("rgb_palette_ratio")

        if c.use_dct_lens:
            for ch in ("r", "g", "b"):
                for tag in ("mean", "std", "skew", "kurt", "energy", "grad"):
                    self._names.append(f"dct_{ch}_{tag}")
                for G in c.grid_sizes:
                    for tag in ("peak", "base", "peak_minus_base"):
                        self._names.append(f"dct_{ch}_g{G}_{tag}")
                for i in range(c.n_radial_bins):
                    self._names.append(f"dct_{ch}_radial_{i:02d}")

        if c.use_qft_lens:
            for tag in ("mean", "std", "skew", "kurt", "energy", "grad"):
                self._names.append(f"qft_{tag}")
            for G in c.grid_sizes:
                for tag in ("peak", "base", "peak_minus_base"):
                    self._names.append(f"qft_g{G}_{tag}")
            for i in range(c.n_radial_bins):
                self._names.append(f"qft_radial_{i:02d}")

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
        rgb = _load_and_normalise(image, c.image_size)
        residual = _compute_residual(rgb, c.residual_sigma)

        blocks: list[np.ndarray] = []
        if c.use_rgb_lens:
            blocks.append(rgb_lens_features(residual))
            blocks.append(palette_ratio(rgb))
        if c.use_dct_lens:
            blocks.append(
                dct_lens_features(
                    residual, c.grid_sizes, c.peak_bin_radius, c.n_radial_bins
                )
            )
        if c.use_qft_lens:
            blocks.append(
                qft_lens_features(
                    residual,
                    c.grid_sizes,
                    c.peak_bin_radius,
                    c.n_radial_bins,
                    c.qft_low_freq_fraction,
                )
            )

        return np.concatenate(blocks).astype(np.float32)

    def extract_batch(self, images) -> np.ndarray:
        """Returns (N, n_features) array."""
        return np.stack([self.extract(im) for im in images], axis=0)