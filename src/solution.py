from pathlib import Path
from typing import List
import argparse
import sys
import joblib
import numpy as np

from feature_extractor import Stage1FeatureExtractor


def load_model(model_path: Path):
    return joblib.load(model_path)


# Global feature extractor instance (initialized once)
_extractor = None


def get_extractor() -> Stage1FeatureExtractor:
    global _extractor
    if _extractor is None:
        _extractor = Stage1FeatureExtractor()
    return _extractor


def classify_images(
    img_paths: List[Path],
    model_path: Path,
) -> List[int]:
    model = load_model(model_path)
    extractor = get_extractor()
    features = extractor.extract_batch(img_paths)
    print(f"Features shape: {features.shape}")
    preds = model.predict(features)
    return preds.tolist()


def collect_images(img_dir: Path) -> List[Path]:
    if not img_dir.is_dir():
        raise NotADirectoryError(f"{img_dir} is not a directory")
    paths = sorted(img_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No .png files found in {img_dir}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a directory of PNG images with a trained model."
    )
    parser.add_argument(
        "img_dir",
        type=Path,
        help="Path to a directory containing .png images.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model.joblib"),
        help="Path to the joblib model file (default: ./model.joblib).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    img_paths = collect_images(args.img_dir)
    preds = classify_images(img_paths, args.model)
    for path, pred in zip(img_paths, preds):
        print(f"{path.name}\t{pred}")
    return 0


if __name__ == "__main__":
    sys.exit(main())