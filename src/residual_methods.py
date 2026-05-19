"""
Residual computation methods shared by the spectral and forensic extractors.

A "residual" here is the high-frequency / noise component of an image with
content suppressed. It is the input to all spectral lenses, and is an
optional pre-processing step for the LBP / wavelet lenses in the forensic
extractor.

Four methods are available; all are pure numpy + scipy:
    'gaussian'        -- input - Gaussian(input)             (cheap, leaky)
    'median'          -- input - Median(input)               (edge-preserving)
    'multi_gaussian'  -- average of Gaussian residuals at multiple sigmas
    'wavelet'         -- Haar wavelet shrinkage denoiser; closest in spirit
                         to classical PRNU work

Each method operates on a single 2D channel (H, W). Callers that need
per-channel residuals on an RGB image should iterate over channels.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage


VALID_METHODS = ("gaussian", "median", "multi_gaussian", "wavelet")


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------

def residual_gaussian(channel: np.ndarray, sigma: float) -> np.ndarray:
    return channel - ndimage.gaussian_filter(channel, sigma=sigma)


def residual_median(channel: np.ndarray, size: int) -> np.ndarray:
    return channel - ndimage.median_filter(channel, size=size)


def residual_multi_gaussian(channel: np.ndarray,
                            sigmas: Sequence[float]) -> np.ndarray:
    """Average of Gaussian residuals across multiple scales."""
    acc = np.zeros_like(channel, dtype=np.float32)
    for s in sigmas:
        acc += channel - ndimage.gaussian_filter(channel, sigma=s)
    return acc / float(len(sigmas))


# --- Haar wavelet shrinkage ------------------------------------------------

def _haar_step(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One level 2D Haar DWT. Returns (cA, cH, cV, cD) on the trimmed grid."""
    h, w = x.shape
    h -= h % 2
    w -= w % 2
    x = x[:h, :w]
    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]
    cA = (a + b + c + d) * 0.5
    cH = (a + b - c - d) * 0.5
    cV = (a - b + c - d) * 0.5
    cD = (a - b - c + d) * 0.5
    return cA, cH, cV, cD


def _haar_istep(cA: np.ndarray, cH: np.ndarray,
                cV: np.ndarray, cD: np.ndarray) -> np.ndarray:
    """Inverse one level 2D Haar DWT."""
    a = (cA + cH + cV + cD) * 0.5
    b = (cA + cH - cV - cD) * 0.5
    c = (cA - cH + cV - cD) * 0.5
    d = (cA - cH - cV + cD) * 0.5
    h2, w2 = a.shape
    out = np.empty((h2 * 2, w2 * 2), dtype=a.dtype)
    out[0::2, 0::2] = a
    out[0::2, 1::2] = b
    out[1::2, 0::2] = c
    out[1::2, 1::2] = d
    return out


def residual_wavelet(channel: np.ndarray, levels: int,
                     lam: float) -> np.ndarray:
    """Haar wavelet shrinkage residual.

    Noise sigma is estimated robustly from the finest diagonal (HH) sub-band
    via MAD/0.6745, then detail sub-bands at every level are soft-thresholded
    at lam * sigma. The denoised reconstruction is subtracted from the input.
    """
    work = channel.astype(np.float32, copy=True)
    cA = work
    details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(levels):
        cA, cH, cV, cD = _haar_step(cA)
        details.append((cH, cV, cD))

    fine_hh = details[0][2]
    sigma_n = float(np.median(np.abs(fine_hh))) / 0.6745
    threshold = lam * sigma_n

    shrunk = []
    for cH, cV, cD in details:
        shrunk.append((
            np.sign(cH) * np.maximum(np.abs(cH) - threshold, 0.0),
            np.sign(cV) * np.maximum(np.abs(cV) - threshold, 0.0),
            np.sign(cD) * np.maximum(np.abs(cD) - threshold, 0.0),
        ))

    rec = cA
    for cH, cV, cD in reversed(shrunk):
        rec = _haar_istep(rec, cH, cV, cD)

    h, w = channel.shape
    if rec.shape != (h, w):
        # Forward trimmed an odd-sized edge; pad reconstruction with the
        # original edge pixels so residual is zero there (no spurious signal).
        full = np.empty((h, w), dtype=rec.dtype)
        rh, rw = rec.shape
        full[:rh, :rw] = rec
        full[rh:, :] = channel[rh:, :]
        full[:rh, rw:] = channel[:rh, rw:]
        rec = full
    return channel - rec


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def compute_residual_channel(
    channel: np.ndarray,
    method: str,
    *,
    sigma: float = 1.0,
    sigmas: Sequence[float] = (0.5, 1.0, 2.0),
    median_size: int = 3,
    wavelet_levels: int = 2,
    wavelet_lambda: float = 3.0,
) -> np.ndarray:
    """Compute a residual for a single 2D channel.

    All method-specific kwargs are accepted; unused ones are ignored. This
    keeps both extractor configs free to expose only the knobs they want.
    """
    if method == "gaussian":
        return residual_gaussian(channel, sigma)
    if method == "median":
        return residual_median(channel, median_size)
    if method == "multi_gaussian":
        return residual_multi_gaussian(channel, sigmas)
    if method == "wavelet":
        return residual_wavelet(channel, wavelet_levels, wavelet_lambda)
    raise ValueError(
        f"Unknown residual method: {method!r}. "
        f"Expected one of {VALID_METHODS}."
    )


def compute_residual_rgb(rgb: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """Per-channel residual on an (H, W, C) image."""
    res = np.empty_like(rgb)
    for c in range(rgb.shape[-1]):
        res[..., c] = compute_residual_channel(rgb[..., c], method, **kwargs)
    return res
