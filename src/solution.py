from pathlib import Path
from typing import List, Tuple
import argparse
import sys
import joblib
import numpy as np

from extractor_factory import build_extractor


# ---------------------------------------------------------------------------
# Hard-coded model path (resolved relative to this script's location, so it
# works regardless of the directory the script is invoked from).
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hierarchical.joblib"


# ---------------------------------------------------------------------------
# Fine class names (index matches the labels the model emits)
# ---------------------------------------------------------------------------

FINE_NAMES = [
    "Real",         # 0
    "HMAR_d20",     # 1
    "HMAR_d30",     # 2
    "LlamaGen_B",   # 3
    "LlamaGen_L",   # 4
    "VAR_d20",      # 5
    "VAR_d30",      # 6
    "RAR_L",        # 7
    "RAR_XXL",      # 8
]


# ---------------------------------------------------------------------------
# Model loading: auto-detect flat Pipeline vs hierarchical bundle
# ---------------------------------------------------------------------------

def load_model(model_path: Path) -> Tuple[object, bool]:
    """Load a saved model.

    Returns (model_obj, is_hierarchical):
      - flat Pipeline   -> (Pipeline, False)
      - bundle dict     -> (dict,     True)
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Expected the trained hierarchical bundle at this location."
        )
    obj = joblib.load(model_path)
    is_hierarchical = isinstance(obj, dict) and "stage1" in obj
    return obj, is_hierarchical


# ---------------------------------------------------------------------------
    # Hierarchical inference (soft routing)
# ---------------------------------------------------------------------------

def predict_fine_soft(bundle: dict, X: np.ndarray) -> np.ndarray:
    """Soft-routed hierarchical prediction.

    Construct a 9-way distribution
        P(fine = f) = P(coarse = family(f)) * P(fine = f | coarse = family(f))
    and argmax. This recovers samples that stage-1 would otherwise mis-route
    whenever the correct family's stage-2 head assigns a much higher
    within-family probability than the wrong (low-confidence) family does.
    Falls back to a hard prediction inside a family if its head does not
    expose predict_proba.
    """
    stage1 = bundle["stage1"]
    heads = bundle["stage2"]
    coarse_to_fine = bundle["coarse_to_fine"]
    n_fine = len(bundle["fine_names"])

    if not hasattr(stage1, "predict_proba"):
        raise RuntimeError(
            "Soft routing requires stage1 to expose predict_proba. "
            "Use a calibrated classifier (LogReg/calibrated SVC/HGB)."
        )

    P_coarse = stage1.predict_proba(X).astype(np.float32)
    stage1_classes = np.asarray(stage1.classes_, dtype=np.int64)
    coarse_col = {int(c): i for i, c in enumerate(stage1_classes)}

    P_fine = np.zeros((len(X), n_fine), dtype=np.float32)
    for coarse_idx, fine_members in coarse_to_fine.items():
        if coarse_idx not in coarse_col:
            continue
        p_c = P_coarse[:, coarse_col[coarse_idx]]
        if len(fine_members) == 1:
            P_fine[:, fine_members[0]] += p_c
            continue
        if coarse_idx not in heads:
            share = p_c / float(len(fine_members))
            for f in fine_members:
                P_fine[:, f] += share
            continue
        head = heads[coarse_idx]
        if hasattr(head, "predict_proba"):
            P_within = head.predict_proba(X).astype(np.float32)
            head_classes = np.asarray(head.classes_, dtype=np.int64)
            for i, fc in enumerate(head_classes):
                P_fine[:, int(fc)] += p_c * P_within[:, i]
        else:
            pred = head.predict(X).astype(np.int64)
            for f in fine_members:
                P_fine[pred == f, f] += p_c[pred == f]

    return P_fine.argmax(axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# Feature extractor singleton
# ---------------------------------------------------------------------------

_extractor = None


def get_extractor():
    """Return the extractor selected by extractor_config.yaml (cached)."""
    global _extractor
    if _extractor is None:
        _extractor = build_extractor()
    return _extractor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_images(img_paths: List[Path]) -> List[int]:
    """Classify a list of images. Returns fine-class labels (0..8).

    Uses the hard-coded MODEL_PATH (models/hierarchical.joblib next to this
    script).
    """
    if not img_paths:
        return []

    model, is_hierarchical = load_model(MODEL_PATH)
    extractor = get_extractor()
    features = extractor.extract_batch([str(p) for p in img_paths])
    print(f"Features shape: {features.shape}  "
          f"(model: {'hierarchical bundle' if is_hierarchical else 'flat pipeline'})",
          file=sys.stderr)

    if is_hierarchical:
        preds = predict_fine_soft(model, features)
    else:
        preds = model.predict(features)

    return [int(p) for p in preds]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_images(img_dir: Path) -> List[Path]:
    if not img_dir.is_dir():
        raise NotADirectoryError(f"{img_dir} is not a directory")
    paths = sorted(img_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No .png files found in {img_dir}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a directory of PNG images using the bundled "
                    "two-stage model at models/hierarchical.joblib."
    )
    parser.add_argument(
        "img_dir",
        type=Path,
        help="Path to a directory containing .png images.",
    )
    parser.add_argument(
        "--names",
        action="store_true",
        help="Also print the human-readable class name next to each label.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    img_paths = collect_images(args.img_dir)
    preds = classify_images(img_paths)
    for path, pred in zip(img_paths, preds):
        if args.names:
            print(f"{path.name}\t{pred}\t{FINE_NAMES[pred]}")
        else:
            print(f"{path.name}\t{pred}")
    return 0


if __name__ == "__main__":
    sys.exit(main())