"""
Lens-level forensic feature extractor.

Four pure-numpy feature families that target *positive camera provenance* --
they characterise "real photo" by the presence of physical capture traces
(heteroscedastic sensor noise, demosaicing grid, decoder-upsampling
signatures, lens chromatic aberration) rather than by the absence of
generative artefacts. Designed to raise the real-class recall of a
multi-class real-vs-AR classifier whose existing residual features are
shrinkage-based (Gaussian, Haar wavelet).

    1. NLF  -- Noise-Level Function shape.
               Per-patch (mean intensity, residual std) pairs, binned by
               intensity, fit to the Poisson-Gaussian model sigma^2 = a*I + b.
               Real photos show a monotonic, well-fit curve; AR/diffusion
               decoders are flat or unstructured.
               Refs: Foi et al. IEEE TIP 17(10), 2008; Liu et al. IEEE
               TPAMI 30(2), 2008.

    2. CFA  -- Colour Filter Array periodicity.
               Each channel's high-pass residual is FFT'd; the magnitude at
               (pi/2, 0), (0, pi/2), (pi/2, pi/2) measures the 2-pixel
               demosaicing periodicity. Real CFA-demosaiced photos show
               strong peaks; AR outputs do not.
               Refs: Popescu & Farid IEEE TSP 53(10), 2005; Gallagher &
               Chen CVPRW 2008.

    3. RES  -- Radial spectrum of the shrinkage residual.
               Same residual extractor the rest of the pipeline uses (re-used
               from residual_methods.py) is FFT'd; the log-spaced 1D radial
               profile of the FFT magnitude captures decoder-upsampling
               replicas that show up as periodic peaks in AR/diffusion
               outputs and are absent (smooth fall-off) in real photos.
               Refs: Durall, Keuper, Keuper CVPR 2020; Corvi et al. CVPRW
               2023.

    4. LCA  -- Lateral chromatic aberration.
               Per-tile sub-pixel cross-correlation finds the (dx, dy) shift
               of R vs G and B vs G; for real photos the radial component
               grows linearly with distance from the optical centre. We
               summarise each channel pair by the fitted slope, R^2, mean
               tangential magnitude and std of the radial component.
               Refs: Johnson & Farid ACM MM&Sec 2006; Mayer & Stamm ICASSP
               2016.

Pure numpy / scipy / PIL -- no torch, no scikit-image, no pywavelets.
Public surface matches the other extractors (feature_names, n_features,
extract, extract_batch).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Union

import numpy as np
from PIL import Image
from scipy import ndimage

from residual_methods import compute_residual_channel, VALID_METHODS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LensFeaturesConfig:
    image_size: int = 256

    # ---- Block toggles --------------------------------------------------
    use_nlf: bool = True
    use_cfa: bool = True
    use_radial_residual: bool = True
    use_angular_residual: bool = True
    use_lca: bool = True

    # ---- NLF -----------------------------------------------------------
    # Patch size for local (mean, std) estimation. 8 is the standard
    # Foi/Liu choice for 256x256 images.
    nlf_patch: int = 8
    # Drop the fraction of patches with highest local gradient magnitude
    # before binning -- those are content, not noise.
    nlf_texture_drop: float = 0.5
    # Number of intensity bins; we bin patches by mean intensity then take
    # the median std per bin to get the NLF curve.
    nlf_n_bins: int = 16

    # ---- CFA -----------------------------------------------------------
    # Sigma of the Gaussian used to extract each channel's high-pass.
    cfa_highpass_sigma: float = 1.0
    # Half-window (in spectrum bins) over which to average around each
    # diagnostic frequency.
    cfa_peak_radius: int = 2

    # ---- Radial residual spectrum --------------------------------------
    # Which residual extractor to apply (must be one of residual_methods.
    # VALID_METHODS). Defaults match the existing spectral extractor.
    res_method: str = "wavelet"
    res_sigma: float = 1.0
    res_sigmas: Sequence[float] = (0.5, 1.0, 2.0)
    res_median_size: int = 3
    res_wavelet_levels: int = 2
    res_wavelet_lambda: float = 3.0
    # Log-spaced bins covering [1, N/2-1] cycles in the 2D FFT.
    res_n_radial_bins: int = 64

    # ---- Angular residual spectrum -------------------------------------
    # Bins the same FFT magnitude (folded to [0, pi) via conjugate symmetry)
    # by orientation rather than radius. Discriminates synthetic-vs-synthetic
    # (each decoder's upsampling kernel leaves a different angular signature)
    # where the radial profile primarily discriminates real-vs-synthetic.
    # Confined to a mid-frequency annulus where upsampling artefacts live.
    ang_n_bins: int = 16
    ang_r_min_frac: float = 0.25
    ang_r_max_frac: float = 0.75

    # ---- LCA -----------------------------------------------------------
    # Tile grid (LCA_TILES x LCA_TILES tiles across the image).
    lca_tiles: int = 4
    # Search radius (pixels) for the per-tile cross-correlation.
    lca_search_radius: int = 3


# ---------------------------------------------------------------------------
# Image I/O (mirrors the other extractors)
# ---------------------------------------------------------------------------

ImageLike = Union[str, "np.ndarray", Image.Image]


def _load_rgb(image: ImageLike, size: int) -> np.ndarray:
    """Return float32 RGB image (size, size, 3) in [0, 1]."""
    if isinstance(image, Image.Image):
        img = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    else:
        img = Image.open(image).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


# ---------------------------------------------------------------------------
# Block 1: Noise-Level Function
# ---------------------------------------------------------------------------

_NLF_FIT_TAGS = ("a", "b", "r2", "valid_frac")
_NLF_CURVE_TAG = "curve"


def _patch_means_stds(gray: np.ndarray, patch: int
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-patch (mean, residual_std, local_gradient_mag) on a non-overlap grid."""
    H, W = gray.shape
    bh = H // patch
    bw = W // patch
    if bh == 0 or bw == 0:
        return (np.empty(0, np.float32),) * 3

    trimmed = gray[:bh * patch, :bw * patch]
    blocks = trimmed.reshape(bh, patch, bw, patch).swapaxes(1, 2)
    # blocks.shape == (bh, bw, patch, patch)
    means = blocks.mean(axis=(2, 3))

    # Residual = patch minus its mean -- crude local high-pass that captures
    # noise without committing to a denoiser model.
    centred = blocks - means[..., None, None]
    stds = centred.std(axis=(2, 3))

    # Local gradient magnitude per patch -- used to filter out textured patches.
    gx = np.diff(trimmed, axis=1, prepend=trimmed[:, :1])
    gy = np.diff(trimmed, axis=0, prepend=trimmed[:1, :])
    gmag = np.sqrt(gx * gx + gy * gy)
    gmag_blocks = gmag[:bh * patch, :bw * patch].reshape(
        bh, patch, bw, patch
    ).swapaxes(1, 2)
    grads = gmag_blocks.mean(axis=(2, 3))

    return means.ravel(), stds.ravel(), grads.ravel()


def _nlf_features(gray: np.ndarray, cfg: LensFeaturesConfig) -> np.ndarray:
    means, stds, grads = _patch_means_stds(gray, cfg.nlf_patch)
    n_bins = cfg.nlf_n_bins

    if means.size == 0:
        return np.zeros(n_bins + len(_NLF_FIT_TAGS), dtype=np.float32)

    # Drop the texturedest fraction -- those patches carry content, not noise.
    keep = grads <= np.quantile(grads, 1.0 - cfg.nlf_texture_drop)
    means_k = means[keep]
    stds_k = stds[keep]
    valid_frac = float(keep.mean())

    if means_k.size < 2 * n_bins:
        return np.concatenate([
            np.zeros(n_bins, dtype=np.float32),
            np.zeros(len(_NLF_FIT_TAGS), dtype=np.float32),
        ])

    # Bin by intensity. Use percentile edges so each bin has ~equal pop.
    edges = np.quantile(means_k, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1e-6  # so the max value lands in the last bin
    curve = np.zeros(n_bins, dtype=np.float32)
    bin_centres = np.zeros(n_bins, dtype=np.float32)
    valid_bins = np.zeros(n_bins, dtype=bool)
    for i in range(n_bins):
        mask = (means_k >= edges[i]) & (means_k < edges[i + 1])
        if mask.sum() >= 3:
            curve[i] = float(np.median(stds_k[mask]))
            bin_centres[i] = float(0.5 * (edges[i] + edges[i + 1]))
            valid_bins[i] = True

    # Poisson-Gaussian fit: sigma^2 = a*I + b. Fit only on valid bins.
    if valid_bins.sum() >= 3:
        x = bin_centres[valid_bins].astype(np.float64)
        y = (curve[valid_bins].astype(np.float64)) ** 2
        A = np.vstack([x, np.ones_like(x)]).T
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        y_hat = a * x + b
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
    else:
        a, b, r2 = 0.0, 0.0, 0.0

    fit = np.asarray([a, b, r2, valid_frac], dtype=np.float32)
    return np.concatenate([curve, fit]).astype(np.float32)


# ---------------------------------------------------------------------------
# Block 2: CFA periodicity
# ---------------------------------------------------------------------------

_CFA_TAGS = ("peak_hh", "peak_vh", "peak_dh", "peak_pool")
# hh = horizontal half-Nyquist (pi/2 along x, 0 along y)
# vh = vertical   half-Nyquist (0 along x, pi/2 along y)
# dh = diagonal   half-Nyquist (pi/2, pi/2)
# pool = max of the three normalised by surrounding spectrum


def _channel_cfa_peaks(channel: np.ndarray, sigma: float,
                       peak_radius: int) -> np.ndarray:
    """Strength of the three CFA-diagnostic peaks in the channel's FFT."""
    # High-pass each channel; this is the residual the demosaicing
    # periodicity lives in.
    hp = channel - ndimage.gaussian_filter(channel, sigma=sigma)
    H, W = hp.shape
    spec = np.abs(np.fft.fftshift(np.fft.fft2(hp - hp.mean())))

    cy, cx = H // 2, W // 2
    qy = H // 4  # offset from centre to half-Nyquist along y
    qx = W // 4  # offset from centre to half-Nyquist along x

    def _peak(y0: int, x0: int) -> float:
        r = peak_radius
        y1 = max(0, y0 - r)
        y2 = min(H, y0 + r + 1)
        x1 = max(0, x0 - r)
        x2 = min(W, x0 + r + 1)
        return float(spec[y1:y2, x1:x2].max())

    # Diagnostic locations on the shifted spectrum.
    p_hh = _peak(cy, cx + qx)        # (0, +pi/2)
    p_vh = _peak(cy + qy, cx)        # (+pi/2, 0)
    p_dh = _peak(cy + qy, cx + qx)   # (+pi/2, +pi/2)
    pool = max(p_hh, p_vh, p_dh)

    # Normalise by the surrounding spectrum energy to make the score
    # invariant to overall residual amplitude.
    denom = float(spec.mean()) + 1e-8
    return np.asarray([p_hh, p_vh, p_dh, pool], dtype=np.float32) / denom


def _cfa_features(rgb: np.ndarray, cfg: LensFeaturesConfig) -> np.ndarray:
    feats = []
    for c in range(3):
        feats.append(_channel_cfa_peaks(
            rgb[..., c], cfg.cfa_highpass_sigma, cfg.cfa_peak_radius,
        ))
    return np.concatenate(feats).astype(np.float32)


# ---------------------------------------------------------------------------
# Block 3: Radial spectrum of the existing shrinkage residual
# ---------------------------------------------------------------------------

def _radial_spectrum(residual: np.ndarray, n_bins: int) -> np.ndarray:
    """Log-spaced 1D radial profile of the 2D FFT magnitude. L1-normalised."""
    H, W = residual.shape
    r = residual - residual.mean()
    spec = np.abs(np.fft.fftshift(np.fft.fft2(r)))

    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    r_max = float(min(cy, cx) - 1)
    if r_max < 2.0:
        return np.zeros(n_bins, dtype=np.float32)

    edges = np.logspace(np.log10(1.0), np.log10(r_max), n_bins + 1)
    profile = np.empty(n_bins, dtype=np.float32)
    rflat = radius.ravel()
    sflat = spec.ravel().astype(np.float32)
    for i in range(n_bins):
        mask = (rflat >= edges[i]) & (rflat < edges[i + 1])
        profile[i] = sflat[mask].mean() if mask.any() else 0.0
    s = profile.sum()
    if s > 0:
        profile = profile / s
    return profile


def _shrinkage_residual(gray: np.ndarray,
                        cfg: LensFeaturesConfig) -> np.ndarray:
    """Compute the configured shrinkage residual on a 2D channel."""
    if cfg.res_method not in VALID_METHODS:
        raise ValueError(
            f"Unknown res_method {cfg.res_method!r}; expected one of {VALID_METHODS}."
        )
    return compute_residual_channel(
        gray, cfg.res_method,
        sigma=cfg.res_sigma,
        sigmas=cfg.res_sigmas,
        median_size=cfg.res_median_size,
        wavelet_levels=cfg.res_wavelet_levels,
        wavelet_lambda=cfg.res_wavelet_lambda,
    )


def _angular_spectrum(residual: np.ndarray, n_bins: int,
                      r_min_frac: float, r_max_frac: float) -> np.ndarray:
    """Orientation profile of the 2D FFT magnitude inside a mid-freq annulus.

    Folds [-pi, pi) to [0, pi) using the conjugate-symmetry of the FFT of a
    real image, so 16 bins resolve to 11.25 deg each. L1-normalised; the
    feature is the angular *shape* of the spectrum, not absolute energy.
    """
    H, W = residual.shape
    r = residual - residual.mean()
    spec = np.abs(np.fft.fftshift(np.fft.fft2(r))).astype(np.float32)

    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    dy = (y - cy).astype(np.float32)
    dx = (x - cx).astype(np.float32)
    radius = np.sqrt(dy * dy + dx * dx)
    # Fold conjugate symmetry to [0, pi) via modulo. atan2(0,0) = 0 is fine
    # since we mask out the DC pixel below.
    angle = np.arctan2(dy, dx) % np.pi

    r_nyq = float(min(cy, cx) - 1)
    r_min = r_min_frac * r_nyq
    r_max = r_max_frac * r_nyq
    if r_max < r_min + 1.0:
        return np.zeros(n_bins, dtype=np.float32)

    mask = (radius >= r_min) & (radius <= r_max)
    a = angle[mask]
    s = spec[mask]
    if a.size == 0:
        return np.zeros(n_bins, dtype=np.float32)

    # np.histogram with weights is the vectorised binning; divide by counts
    # to get the mean per bin (matches the radial-spectrum convention).
    weight_sum, _ = np.histogram(a, bins=n_bins, range=(0.0, np.pi), weights=s)
    counts, _ = np.histogram(a, bins=n_bins, range=(0.0, np.pi))
    profile = np.where(counts > 0,
                       weight_sum / np.maximum(counts, 1),
                       0.0).astype(np.float32)
    total = float(profile.sum())
    if total > 0.0:
        profile = profile / total
    return profile


def _radial_residual_features(residual: np.ndarray,
                              cfg: LensFeaturesConfig) -> np.ndarray:
    return _radial_spectrum(residual, cfg.res_n_radial_bins)


def _angular_residual_features(residual: np.ndarray,
                               cfg: LensFeaturesConfig) -> np.ndarray:
    return _angular_spectrum(
        residual, cfg.ang_n_bins, cfg.ang_r_min_frac, cfg.ang_r_max_frac,
    )


# ---------------------------------------------------------------------------
# Block 4: Lateral Chromatic Aberration
# ---------------------------------------------------------------------------

_LCA_PAIRS = (("r", 0, 1), ("b", 2, 1))   # (label, channel, ref); ref is G
_LCA_PAIR_TAGS = ("slope", "r2", "mean_tan", "std_rad")


def _best_shift(tile_src: np.ndarray, tile_ref: np.ndarray,
                radius: int) -> tuple[float, float]:
    """Integer (dy, dx) shift maximising NCC between src and ref, in +-radius.

    Returns (0.0, 0.0) if tiles are flat (zero variance).
    """
    s = tile_src - tile_src.mean()
    r = tile_ref - tile_ref.mean()
    s_norm = float(np.sqrt((s * s).sum())) + 1e-8
    r_norm = float(np.sqrt((r * r).sum())) + 1e-8
    if s_norm < 1e-6 or r_norm < 1e-6:
        return 0.0, 0.0

    best = -np.inf
    best_dy = 0
    best_dx = 0
    H, W = s.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # Slice the overlapping region after shifting src by (dy, dx)
            y0_s = max(0, -dy)
            y0_r = max(0, dy)
            x0_s = max(0, -dx)
            x0_r = max(0, dx)
            h = H - abs(dy)
            w = W - abs(dx)
            if h <= 1 or w <= 1:
                continue
            a = s[y0_s:y0_s + h, x0_s:x0_s + w]
            b = r[y0_r:y0_r + h, x0_r:x0_r + w]
            ncc = float((a * b).sum()) / (s_norm * r_norm)
            if ncc > best:
                best = ncc
                best_dy = dy
                best_dx = dx
    return float(best_dy), float(best_dx)


def _lca_pair_features(src: np.ndarray, ref: np.ndarray,
                       tiles: int, radius: int) -> np.ndarray:
    """For a single channel pair, fit radial shift vs distance and summarise."""
    H, W = src.shape
    th = H // tiles
    tw = W // tiles
    cy = H / 2.0
    cx = W / 2.0

    radial = []
    tangent = []
    dist = []
    for ti in range(tiles):
        for tj in range(tiles):
            y0 = ti * th
            x0 = tj * tw
            s = src[y0:y0 + th, x0:x0 + tw]
            r = ref[y0:y0 + th, x0:x0 + tw]
            dy, dx = _best_shift(s, r, radius)
            # Tile centre relative to image centre
            ty = (y0 + th / 2.0) - cy
            tx = (x0 + tw / 2.0) - cx
            d = float(np.hypot(ty, tx))
            if d < 1e-6:
                continue  # central tile -- radial direction undefined
            ny = ty / d
            nx = tx / d
            rad = dy * ny + dx * nx           # along the radial outward direction
            tan = -dy * nx + dx * ny          # perpendicular component
            radial.append(rad)
            tangent.append(tan)
            dist.append(d)

    if len(dist) < 3:
        return np.zeros(len(_LCA_PAIR_TAGS), dtype=np.float32)

    radial = np.asarray(radial, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)

    # Fit radial_shift = slope * distance (no intercept -- LCA passes
    # through zero at the optical centre).
    slope = float((radial * dist).sum() / ((dist * dist).sum() + 1e-12))
    y_hat = slope * dist
    ss_res = float(np.sum((radial - y_hat) ** 2))
    ss_tot = float(np.sum((radial - radial.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    mean_tan = float(np.mean(np.abs(tangent)))
    std_rad = float(np.std(radial))
    return np.asarray([slope, r2, mean_tan, std_rad], dtype=np.float32)


def _lca_features(rgb: np.ndarray, cfg: LensFeaturesConfig) -> np.ndarray:
    g = rgb[..., 1]
    feats = []
    for _label, ch_idx, _ref_idx in _LCA_PAIRS:
        feats.append(_lca_pair_features(
            rgb[..., ch_idx], g, cfg.lca_tiles, cfg.lca_search_radius,
        ))
    return np.concatenate(feats).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LensFeaturesExtractor:
    """NLF + CFA + radial-residual-spectrum + LCA. Pure numpy."""

    def __init__(self, config: LensFeaturesConfig | None = None):
        self.cfg = config or LensFeaturesConfig()
        self._names: list[str] = []
        self._populate_names()

    def _populate_names(self) -> None:
        c = self.cfg
        if c.use_nlf:
            for i in range(c.nlf_n_bins):
                self._names.append(f"nlf_{_NLF_CURVE_TAG}_b{i:02d}")
            for tag in _NLF_FIT_TAGS:
                self._names.append(f"nlf_{tag}")
        if c.use_cfa:
            for ch in ("r", "g", "b"):
                for tag in _CFA_TAGS:
                    self._names.append(f"cfa_{ch}_{tag}")
        if c.use_radial_residual:
            for i in range(c.res_n_radial_bins):
                self._names.append(f"resradspec_b{i:03d}")
        if c.use_angular_residual:
            for i in range(c.ang_n_bins):
                self._names.append(f"resangspec_b{i:02d}")
        if c.use_lca:
            for label, _ch, _ref in _LCA_PAIRS:
                for tag in _LCA_PAIR_TAGS:
                    self._names.append(f"lca_{label}_{tag}")

    @property
    def feature_names(self) -> list[str]:
        return list(self._names)

    @property
    def n_features(self) -> int:
        return len(self._names)

    def extract(self, image: ImageLike) -> np.ndarray:
        c = self.cfg
        rgb = _load_rgb(image, c.image_size)
        gray = _to_gray(rgb)

        # Compute the shrinkage residual once if either spectrum block uses it.
        residual = None
        if c.use_radial_residual or c.use_angular_residual:
            residual = _shrinkage_residual(gray, c)

        blocks: list[np.ndarray] = []
        if c.use_nlf:
            blocks.append(_nlf_features(gray, c))
        if c.use_cfa:
            blocks.append(_cfa_features(rgb, c))
        if c.use_radial_residual:
            blocks.append(_radial_residual_features(residual, c))
        if c.use_angular_residual:
            blocks.append(_angular_residual_features(residual, c))
        if c.use_lca:
            blocks.append(_lca_features(rgb, c))
        return np.concatenate(blocks).astype(np.float32)

    def extract_batch(self, images) -> np.ndarray:
        return np.stack([self.extract(im) for im in images], axis=0)
