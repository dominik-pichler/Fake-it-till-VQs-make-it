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
# Hierarchical inference (hard routing)
# ---------------------------------------------------------------------------

def predict_fine_hard(bundle: dict, X: np.ndarray) -> np.ndarray:
    """Stage-1 picks the family; the chosen stage-2 head picks the fine class.

    For singleton families (Real), the fine prediction is deterministic.
    If a stage-2 head was skipped at train time (degenerate data), falls
    back to the family's first fine member so prediction still terminates.
    """
    stage1 = bundle["stage1"]
    heads = bundle["stage2"]
    coarse_to_fine = bundle["coarse_to_fine"]

    coarse_pred = stage1.predict(X).astype(np.int64)
    fine_pred = np.full(len(X), -1, dtype=np.int64)

    for coarse_idx, fine_members in coarse_to_fine.items():
        mask = coarse_pred == coarse_idx
        if not mask.any():
            continue
        if len(fine_members) == 1:
            fine_pred[mask] = fine_members[0]
        elif coarse_idx in heads:
            fine_pred[mask] = heads[coarse_idx].predict(X[mask]).astype(np.int64)
        else:
            # Head was skipped during training — fall back deterministically
            fine_pred[mask] = fine_members[0]

    if (fine_pred == -1).any():
        unrouted = int((fine_pred == -1).sum())
        raise RuntimeError(
            f"{unrouted} samples were not routed to any family. "
            f"Stage-1 predicted a coarse class not in coarse_to_fine."
        )
    return fine_pred


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
        preds = predict_fine_hard(model, features)
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