
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from feature_extractor import FeatureConfig, Stage1FeatureExtractor


# ---------------------------------------------------------------------------
# Default model path (can be overridden via env var or function parameter)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "best.joblib"


# ---------------------------------------------------------------------------
# Classifier singleton for efficient repeated calls
# ---------------------------------------------------------------------------

class _ClassifierSingleton:
    """Lazy-loaded classifier to avoid reloading model on each call."""

    _instance: "_ClassifierSingleton | None" = None
    _model: Pipeline | None = None
    _extractor: Stage1FeatureExtractor | None = None
    _model_path: Path | None = None

    @classmethod
    def get(cls, model_path: Path | None = None) -> "_ClassifierSingleton":
        if cls._instance is None:
            cls._instance = cls()

        # Reload if different model path requested
        resolved_path = cls._resolve_model_path(model_path)
        if cls._instance._model_path != resolved_path:
            cls._instance._load(resolved_path)

        return cls._instance

    @staticmethod
    def _resolve_model_path(model_path: Path | None) -> Path:
        if model_path is not None:
            return Path(model_path)
        env_path = os.environ.get("DEEPFAKE_MODEL_PATH")
        if env_path:
            return Path(env_path)
        return DEFAULT_MODEL_PATH

    def _load(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Train a model first with: python deep_fake_classifier_pipeline.py train ..."
            )
        self._model = load(model_path)
        self._extractor = Stage1FeatureExtractor(FeatureConfig())
        self._model_path = model_path

    @property
    def model(self) -> Pipeline:
        if self._model is None:
            raise RuntimeError("Classifier not loaded")
        return self._model

    @property
    def extractor(self) -> Stage1FeatureExtractor:
        if self._extractor is None:
            raise RuntimeError("Extractor not loaded")
        return self._extractor


# ---------------------------------------------------------------------------
# Public API: classify_images
# ---------------------------------------------------------------------------

def classify_images(
    img_paths: List[Path],
    model_path: Path | None = None,
) -> List[int]:
    """
    Classify a list of images and return predicted class labels.

    Args:
        img_paths: List of paths to image files (PNG, JPG, etc.)
        model_path: Optional path to trained model (.joblib file).
                    If not provided, uses DEEPFAKE_MODEL_PATH env var
                    or falls back to models/best.joblib

    Returns:
        List of predicted class labels (0=Real, 1=LlamaGen, 2=VAR_HMAR, 3=RAR)
    """
    if not img_paths:
        return []

    classifier = _ClassifierSingleton.get(model_path)

    # Extract features for all images
    features = classifier.extractor.extract_batch([str(p) for p in img_paths])

    # Predict
    predictions = classifier.model.predict(features)

    return [int(p) for p in predictions]


# ---------------------------------------------------------------------------
# Label / folder mapping
# ---------------------------------------------------------------------------

CLASS_NAMES = ["Real", "LlamaGen", "VAR_HMAR", "RAR"]

# Sub-source folder name -> 4-way class index
SUBSOURCE_TO_LABEL = {
    "real":               0,
    "llamagen_B_VQ-16":   1,
    "llamagen_L_VQ-16":   1,
    "hmar_d20":           2,
    "hmar_d30":           2,
    "nspvar_20":          2,
    "nspvar_30":          2,
    "rar_l":              3,
    "rar_xxl":            3,
}


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def collect_split(split_dir: Path) -> tuple[list[Path], np.ndarray, list[str]]:
    """
    Walk a split directory (train/ or val/) and collect image paths + labels.
    Returns (paths, labels, subsource_names).
    """
    paths, labels, subs = [], [], []
    for sub in sorted(split_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name not in SUBSOURCE_TO_LABEL:
            print(f"  [warn] unknown sub-source folder: {sub.name}", file=sys.stderr)
            continue
        label = SUBSOURCE_TO_LABEL[sub.name]
        for p in sorted(sub.glob("*.png")):
            paths.append(p)
            labels.append(label)
            subs.append(sub.name)
    return paths, np.asarray(labels, dtype=np.int64), subs


def collect_test(test_dir: Path) -> list[Path]:
    """Test images are flat: test/00000.png, test/00001.png, ..."""
    return sorted(test_dir.glob("*.png"))


# ---------------------------------------------------------------------------
# Feature extraction with disk cache
# ---------------------------------------------------------------------------

def extract_split(
    paths: list[Path],
    extractor: Stage1FeatureExtractor,
    cache_path: Path,
    log_every: int = 500,
) -> np.ndarray:
    """
    Extract features for a list of image paths. Caches to .npy file.
    If cache exists and shape matches, loads from cache.
    """
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (len(paths), extractor.n_features):
            print(f"  loaded cached features: {cache_path} {cached.shape}")
            return cached
        print(f"  cache shape mismatch, recomputing: {cache_path}")

    feats = np.empty((len(paths), extractor.n_features), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(paths):
        feats[i] = extractor.extract(str(p))
        if (i + 1) % log_every == 0 or i == len(paths) - 1:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (len(paths) - i - 1) / rate
            print(f"  [{i+1:>6}/{len(paths)}] {rate:.1f} img/s  eta={eta/60:.1f}min")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, feats)
    print(f"  saved: {cache_path} {feats.shape}")
    return feats


# ---------------------------------------------------------------------------
# Subcommand: extract -> This runs the feature extraction!
# ---------------------------------------------------------------------------

def cmd_extract(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = Stage1FeatureExtractor(FeatureConfig())
    print(f"Extractor: {extractor.n_features} features")

    # Save feature names + class names once for later inspection
    (out_dir / "feature_names.json").write_text(
        json.dumps(extractor.feature_names, indent=2)
    )
    (out_dir / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2))

    for split in ("train", "val"):
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"[skip] {split_dir} does not exist")
            continue
        print(f"\n=== {split} ===")
        paths, labels, subs = collect_split(split_dir)
        print(f"  {len(paths)} images across {len(set(subs))} sub-sources")

        feats = extract_split(paths, extractor, out_dir / f"{split}_X.npy")
        np.save(out_dir / f"{split}_y.npy", labels)
        (out_dir / f"{split}_paths.json").write_text(
            json.dumps([str(p) for p in paths])
        )
        (out_dir / f"{split}_subs.json").write_text(json.dumps(subs))

    test_dir = data_root / "test"
    if test_dir.exists():
        print("\n=== test ===")
        paths = collect_test(test_dir)
        print(f"  {len(paths)} images")
        extract_split(paths, extractor, out_dir / "test_X.npy")
        (out_dir / "test_paths.json").write_text(
            json.dumps([str(p) for p in paths])
        )


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def build_classifiers(random_state: int = 0) -> dict[str, Pipeline]:
    """All three candidates share a StandardScaler front-end."""
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=1.0,
                random_state=random_state,
            )),
        ]),
        "linear_svm": Pipeline([
            ("scaler", StandardScaler()),
            # Wrap in CalibratedClassifierCV so we get predict_proba.
            # Uses internal CV on the *training* data only; val/ is untouched.
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", max_iter=5000,
                          random_state=random_state),
                cv=3,
            )),
        ]),
        "hgb": Pipeline([
            # Tree models don't need scaling, but keeping the step makes the
            # pipelines uniform and StandardScaler is essentially free here.
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.05,
                max_depth=None,
                l2_regularization=0.0,
                random_state=random_state,
            )),
        ]),
    }


def evaluate(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    pred = pipe.predict(X)
    acc = float((pred == y).mean())
    report = classification_report(
        y, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y, pred).tolist()
    return {"accuracy": acc, "report": report, "confusion": cm}


def cmd_train(args: argparse.Namespace) -> None:
    feat_dir = Path(args.features)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tr = np.load(feat_dir / "train_X.npy")
    y_tr = np.load(feat_dir / "train_y.npy")
    X_va = np.load(feat_dir / "val_X.npy")
    y_va = np.load(feat_dir / "val_y.npy")
    print(f"train: {X_tr.shape}, val: {X_va.shape}")
    print(f"class counts (train): {np.bincount(y_tr).tolist()}")
    print(f"class counts (val):   {np.bincount(y_va).tolist()}")

    classifiers = build_classifiers(random_state=args.seed)
    results: dict[str, dict] = {}

    for name, pipe in classifiers.items():
        print(f"\n--- {name} ---")
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        train_dt = time.time() - t0

        train_eval = evaluate(pipe, X_tr, y_tr)
        val_eval = evaluate(pipe, X_va, y_va)
        results[name] = {
            "fit_seconds": train_dt,
            "train": train_eval,
            "val": val_eval,
        }

        print(f"  fit: {train_dt:.1f}s")
        print(f"  train acc: {train_eval['accuracy']:.4f}")
        print(f"  val   acc: {val_eval['accuracy']:.4f}")
        for cls, m in val_eval["report"].items():
            if cls in CLASS_NAMES:
                print(f"    {cls:10s} P={m['precision']:.3f} "
                      f"R={m['recall']:.3f} F1={m['f1-score']:.3f}")

        dump(pipe, out_dir / f"{name}.joblib")

    # Pick best on val accuracy
    best_name = max(results, key=lambda k: results[k]["val"]["accuracy"])
    best_pipe = classifiers[best_name]
    print(f"\n>>> best on val: {best_name} "
          f"(acc={results[best_name]['val']['accuracy']:.4f})")
    dump(best_pipe, out_dir / "best.joblib")

    (out_dir / "results.json").write_text(json.dumps(
        {"best": best_name, "per_model": results}, indent=2
    ))
    print(f"saved: {out_dir / 'best.joblib'}, {out_dir / 'results.json'}")


# ---------------------------------------------------------------------------
# Subcommand: predict
# ---------------------------------------------------------------------------

def cmd_predict(args: argparse.Namespace) -> None:
    feat_dir = Path(args.features)
    test_X = np.load(feat_dir / "test_X.npy")
    test_paths = json.loads((feat_dir / "test_paths.json").read_text())
    print(f"test: {test_X.shape}")

    pipe: Pipeline = load(args.model)
    print(f"loaded model: {args.model}")

    pred = pipe.predict(test_X)

    # If the model exposes probabilities, include them — useful for ensembling
    proba = pipe.predict_proba(test_X) if hasattr(pipe, "predict_proba") else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        header = ["filename", "label", "class_name"]
        if proba is not None:
            header += [f"p_{c}" for c in CLASS_NAMES]
        f.write(",".join(header) + "\n")

        for i, p in enumerate(test_paths):
            fname = Path(p).name
            row = [fname, str(int(pred[i])), CLASS_NAMES[int(pred[i])]]
            if proba is not None:
                row += [f"{proba[i, k]:.6f}" for k in range(len(CLASS_NAMES))]
            f.write(",".join(row) + "\n")
    print(f"wrote: {out_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="Extract and cache features for all splits")
    e.add_argument("--data-root", required=True,
                   help="Root with train/, val/, test/ subdirs")
    e.add_argument("--out", required=True, help="Where to write *_X.npy etc.")
    e.set_defaults(func=cmd_extract)

    t = sub.add_parser("train", help="Train all three classifiers, pick best on val")
    t.add_argument("--features", required=True, help="Feature cache directory")
    t.add_argument("--out", required=True, help="Where to write *.joblib + results.json")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="Predict on the test set")
    pr.add_argument("--features", required=True)
    pr.add_argument("--model", required=True, help="Path to best.joblib (or any *.joblib)")
    pr.add_argument("--out", required=True, help="Output CSV")
    pr.set_defaults(func=cmd_predict)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()