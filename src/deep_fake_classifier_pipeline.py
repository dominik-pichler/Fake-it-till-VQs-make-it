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

from extractor_factory import build_extractor


# ---------------------------------------------------------------------------
# Default model path (can be overridden via env var or function parameter)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "best.joblib"


# ---------------------------------------------------------------------------
# Label / folder mapping
# ---------------------------------------------------------------------------

# Coarse (stage-1) classes
COARSE_NAMES = ["Real", "LlamaGen", "VAR_HMAR", "RAR"]

# Fine (stage-2) classes -- index must match SUBSOURCE_TO_LABEL values below
FINE_NAMES = [
    "Real",         # 0  (ImageNet)
    "HMAR_d20",     # 1
    "HMAR_d30",     # 2
    "LlamaGen_B",   # 3
    "LlamaGen_L",   # 4
    "VAR_d20",      # 5
    "VAR_d30",      # 6
    "RAR_L",        # 7
    "RAR_XXL",      # 8
]

# Sub-source folder name -> fine class index
SUBSOURCE_TO_LABEL = {
    "real":               0,
    "hmar_d20":           1,
    "hmar_d30":           2,
    "llamagen_B_VQ-16":   3,
    "llamagen_L_VQ-16":   4,
    "nspvar_20":          5,   # VAR-d20
    "nspvar_30":          6,   # VAR-d30
    "rar_l":              7,
    "rar_xxl":            8,
}

# fine_idx -> coarse_idx
#                          0  1  2  3  4  5  6  7  8
#                         Real H  H  L  L  V  V  R  R
FINE_TO_COARSE = np.array([0, 2, 2, 1, 1, 2, 2, 3, 3], dtype=np.int64)

# coarse_idx -> ordered list of fine indices belonging to that family
COARSE_TO_FINE: dict[int, list[int]] = {
    0: [0],            # Real
    1: [3, 4],         # LlamaGen: B, L
    2: [1, 2, 5, 6],   # VAR_HMAR: HMAR_d20, HMAR_d30, VAR_d20, VAR_d30
    3: [7, 8],         # RAR: L, XXL
}

# Kept for backward compatibility with older code paths
CLASS_NAMES = FINE_NAMES


# ---------------------------------------------------------------------------
# Classifier singleton for efficient repeated calls
# ---------------------------------------------------------------------------

class _ClassifierSingleton:
    """Lazy-loaded classifier to avoid reloading model on each call.

    Supports both the legacy flat Pipeline (`best.joblib`) and the new
    hierarchical bundle dict (`hierarchical.joblib`).
    """

    _instance: "_ClassifierSingleton | None" = None
    _model: object | None = None         # Pipeline OR bundle dict
    _extractor: object | None = None     # extractor type chosen by YAML config
    _model_path: Path | None = None
    _is_hierarchical: bool = False

    @classmethod
    def get(cls, model_path: Path | None = None) -> "_ClassifierSingleton":
        if cls._instance is None:
            cls._instance = cls()

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
        obj = load(model_path)
        # A bundle is a dict with "stage1" key; a flat model is a Pipeline.
        self._is_hierarchical = isinstance(obj, dict) and "stage1" in obj
        self._model = obj
        self._extractor = build_extractor()
        self._model_path = model_path

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Classifier not loaded")
        return self._model

    @property
    def is_hierarchical(self) -> bool:
        return self._is_hierarchical

    @property
    def extractor(self):
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
    Classify a list of images and return predicted FINE class labels (0..8).

    See FINE_NAMES for the label -> name mapping.

    Args:
        img_paths: List of paths to image files (PNG, JPG, etc.)
        model_path: Optional path to trained model (.joblib file).
                    Accepts either a flat Pipeline or a hierarchical bundle.
                    If not provided, uses DEEPFAKE_MODEL_PATH env var
                    or falls back to models/best.joblib

    Returns:
        List of predicted fine-grained class labels.
    """
    if not img_paths:
        return []

    classifier = _ClassifierSingleton.get(model_path)
    features = classifier.extractor.extract_batch([str(p) for p in img_paths])

    if classifier.is_hierarchical:
        predictions, _ = predict_fine_hard(classifier.model, features)
    else:
        predictions = classifier.model.predict(features)

    return [int(p) for p in predictions]


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def collect_split(split_dir: Path) -> tuple[list[Path], np.ndarray, list[str]]:
    """
    Walk a split directory (train/ or val/) and collect image paths + FINE labels.
    Returns (paths, fine_labels, subsource_names).
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
# Per-extractor feature directory scoping
# ---------------------------------------------------------------------------

def _extractor_kind() -> str:
    """Active extractor kind from the YAML config (e.g. 'spectral')."""
    from extractor_factory import load_config
    return str(load_config().get("extractor", "spectral"))


def scoped_feature_dir(base: Path | str) -> Path:
    """Return <base>/<extractor_kind> so each extractor caches into its own
    subdirectory. The directory is NOT created here -- callers do that.
    """
    return Path(base) / _extractor_kind()


# ---------------------------------------------------------------------------
# Feature extraction with disk cache
# ---------------------------------------------------------------------------

def extract_split(
    paths: list[Path],
    extractor,
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
    for i, p in enumerate(paths): # For each image get the features.
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
# Subcommand: extract
# ---------------------------------------------------------------------------

def cmd_extract(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    out_dir = scoped_feature_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = build_extractor()
    print(f"Extractor: {type(extractor).__name__} with {extractor.n_features} features")
    print(f"Feature cache: {out_dir}")

    # Save metadata once for later inspection
    (out_dir / "feature_names.json").write_text(
        json.dumps(extractor.feature_names, indent=2)
    )
    (out_dir / "fine_names.json").write_text(json.dumps(FINE_NAMES, indent=2))
    (out_dir / "coarse_names.json").write_text(json.dumps(COARSE_NAMES, indent=2))

    for split in ("train", "val"):
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"[skip] {split_dir} does not exist")
            continue
        print(f"\n=== {split} ===")
        paths, labels, subs = collect_split(split_dir)
        print(f"  {len(paths)} images across {len(set(subs))} sub-sources")

        feats = extract_split(paths, extractor, out_dir / f"{split}_X.npy")
        np.save(out_dir / f"{split}_y.npy", labels)  # fine labels
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
# Classifier factory + evaluation
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
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", max_iter=5000,
                          random_state=random_state),
                cv=3,
            )),
        ]),
        "hgb": Pipeline([
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


def build_stage2_candidates(random_state: int = 0) -> dict[str, Pipeline]:
    """Candidate heads for within-family disambiguation.

    Mirrors build_classifiers() so each family can pick the head that fits
    its (typically smaller, harder) subset best on val.
    """
    return build_classifiers(random_state=random_state)


def evaluate(pipe: Pipeline, X: np.ndarray, y: np.ndarray,
             target_names: list[str]) -> dict:
    pred = pipe.predict(X)
    acc = float((pred == y).mean())
    # Restrict labels to those actually present so the report stays sane
    # even on tiny per-family subsets.
    labels_present = sorted(set(np.unique(y).tolist()) | set(np.unique(pred).tolist()))
    names_present = [target_names[i] for i in labels_present]
    report = classification_report(
        y, pred, labels=labels_present, target_names=names_present,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y, pred, labels=labels_present).tolist()
    return {"accuracy": acc, "report": report, "confusion": cm,
            "labels": labels_present}


# ---------------------------------------------------------------------------
# Stage-2: per-family heads
# ---------------------------------------------------------------------------

def train_stage2_heads(
    X_tr: np.ndarray, y_fine_tr: np.ndarray,
    X_va: np.ndarray, y_fine_va: np.ndarray,
    random_state: int = 0,
) -> tuple[dict[int, Pipeline], dict[int, dict]]:
    """Train one head per multi-member coarse family on GT-routed data.

    Returns:
        heads: dict coarse_idx -> fitted Pipeline
        head_results: dict coarse_idx -> {"train": eval, "val": eval, ...}
    """
    heads: dict[int, Pipeline] = {}
    head_results: dict[int, dict] = {}

    for coarse_idx, fine_members in COARSE_TO_FINE.items():
        if len(fine_members) < 2:
            continue  # singleton family (Real) needs no head

        family_name = COARSE_NAMES[coarse_idx]
        mask_tr = np.isin(y_fine_tr, fine_members)
        mask_va = np.isin(y_fine_va, fine_members)

        Xf_tr, yf_tr = X_tr[mask_tr], y_fine_tr[mask_tr]
        Xf_va, yf_va = X_va[mask_va], y_fine_va[mask_va]

        print(f"\n  --- stage2 head [{family_name}] "
              f"({len(fine_members)}-way) ---")
        print(f"    n_train={mask_tr.sum()}, n_val={mask_va.sum()}")
        tr_counts = {FINE_NAMES[c]: int((yf_tr == c).sum()) for c in fine_members}
        va_counts = {FINE_NAMES[c]: int((yf_va == c).sum()) for c in fine_members}
        print(f"    train class counts: {tr_counts}")
        print(f"    val   class counts: {va_counts}")

        # Guard against degenerate subsets: at least 2 distinct fine classes,
        # each with >=1 sample, must be present in train.
        present_in_train = [c for c in fine_members if (yf_tr == c).sum() > 0]
        if len(present_in_train) < 2:
            missing = [FINE_NAMES[c] for c in fine_members
                       if (yf_tr == c).sum() == 0]
            print(f"    [skip] head not trainable: only "
                  f"{len(present_in_train)} of {len(fine_members)} fine "
                  f"classes have train data. Missing: {missing}", file=sys.stderr)
            head_results[coarse_idx] = {
                "family_name": family_name,
                "fine_members": fine_members,
                "skipped_reason": f"missing train data for {missing}",
                "n_train": int(mask_tr.sum()),
                "n_val": int(mask_va.sum()),
            }
            continue

        candidates = build_stage2_candidates(random_state=random_state)
        per_candidate: dict[str, dict] = {}
        best_name: str | None = None
        best_score: float = -1.0
        best_pipe: Pipeline | None = None
        selection_basis = "val" if mask_va.sum() > 0 else "train"

        for cand_name, pipe in candidates.items():
            t0 = time.time()
            pipe.fit(Xf_tr, yf_tr)
            fit_dt = time.time() - t0

            tr_eval = evaluate(pipe, Xf_tr, yf_tr, FINE_NAMES)
            va_eval = (evaluate(pipe, Xf_va, yf_va, FINE_NAMES)
                       if mask_va.sum() > 0 else None)
            score = va_eval["accuracy"] if va_eval else tr_eval["accuracy"]
            va_acc_str = f"{va_eval['accuracy']:.4f}" if va_eval else "n/a"

            print(f"    [{cand_name:10s}] fit: {fit_dt:.1f}s  "
                  f"train acc={tr_eval['accuracy']:.4f}  val acc={va_acc_str}")
            if va_eval:
                for cls_idx in fine_members:
                    cls_name = FINE_NAMES[cls_idx]
                    m = va_eval["report"].get(cls_name)
                    if m:
                        print(f"        {cls_name:12s} "
                              f"P={m['precision']:.3f} R={m['recall']:.3f} "
                              f"F1={m['f1-score']:.3f}")

            per_candidate[cand_name] = {
                "fit_seconds": fit_dt,
                "train": tr_eval,
                "val": va_eval,
            }
            if score > best_score:
                best_score = score
                best_name = cand_name
                best_pipe = pipe

        assert best_pipe is not None and best_name is not None
        print(f"    >>> best for {family_name}: {best_name} "
              f"({selection_basis} acc={best_score:.4f})")

        heads[coarse_idx] = best_pipe
        head_results[coarse_idx] = {
            "family_name": family_name,
            "fine_members": fine_members,
            "n_train": int(mask_tr.sum()),
            "n_val": int(mask_va.sum()),
            "best_candidate": best_name,
            "selection_basis": selection_basis,
            "selection_score": best_score,
            "per_candidate": per_candidate,
        }

    return heads, head_results


# ---------------------------------------------------------------------------
# Hierarchical inference (hard routing)
# ---------------------------------------------------------------------------

def predict_fine_hard(
    bundle: dict, X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Hard-routed hierarchical prediction.

    Stage-1 picks the coarse family; the corresponding stage-2 head (if any)
    picks the fine class. For singleton families (Real), the fine class is
    deterministic.

    Returns:
        fine_pred: (N,) int64 array of fine-class indices
        coarse_pred: (N,) int64 array of coarse-class indices
    """
    stage1: Pipeline = bundle["stage1"]
    heads: dict[int, Pipeline] = bundle["stage2"]
    coarse_to_fine: dict[int, list[int]] = bundle["coarse_to_fine"]

    coarse_pred = stage1.predict(X).astype(np.int64)
    fine_pred = np.full(len(X), -1, dtype=np.int64)

    for coarse_idx, fine_members in coarse_to_fine.items():
        mask = coarse_pred == coarse_idx
        if not mask.any():
            continue
        if len(fine_members) == 1:
            fine_pred[mask] = fine_members[0]
        elif coarse_idx in heads:
            head = heads[coarse_idx]
            fine_pred[mask] = head.predict(X[mask]).astype(np.int64)
        else:
            # Stage-2 head was skipped (e.g. degenerate training subset).
            # Fall back to the family's first fine member so prediction
            # still terminates; flag this so callers can see it happened.
            fine_pred[mask] = fine_members[0]

    # Safety check: every sample should have been assigned
    if (fine_pred == -1).any():
        unrouted = int((fine_pred == -1).sum())
        raise RuntimeError(
            f"{unrouted} samples were not routed to any family. "
            f"This means stage-1 predicted a coarse class not in coarse_to_fine."
        )
    return fine_pred, coarse_pred


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    feat_dir = scoped_feature_dir(args.features)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Feature cache: {feat_dir}")

    X_tr = np.load(feat_dir / "train_X.npy")
    y_fine_tr = np.load(feat_dir / "train_y.npy")
    X_va = np.load(feat_dir / "val_X.npy")
    y_fine_va = np.load(feat_dir / "val_y.npy")
    print(f"train: {X_tr.shape}, val: {X_va.shape}")
    print(f"fine class counts (train): {np.bincount(y_fine_tr, minlength=len(FINE_NAMES)).tolist()}")
    print(f"fine class counts (val):   {np.bincount(y_fine_va, minlength=len(FINE_NAMES)).tolist()}")

    # Derive coarse labels for stage-1
    y_coarse_tr = FINE_TO_COARSE[y_fine_tr]
    y_coarse_va = FINE_TO_COARSE[y_fine_va]
    print(f"coarse class counts (train): {np.bincount(y_coarse_tr, minlength=len(COARSE_NAMES)).tolist()}")
    print(f"coarse class counts (val):   {np.bincount(y_coarse_va, minlength=len(COARSE_NAMES)).tolist()}")

    # ---- Stage 1: train candidates on coarse labels, pick best on val ----
    classifiers = build_classifiers(random_state=args.seed)
    results: dict[str, dict] = {}

    for name, pipe in classifiers.items():
        print(f"\n--- stage1 [{name}] ---")
        t0 = time.time()
        pipe.fit(X_tr, y_coarse_tr)
        train_dt = time.time() - t0

        train_eval = evaluate(pipe, X_tr, y_coarse_tr, COARSE_NAMES)
        val_eval = evaluate(pipe, X_va, y_coarse_va, COARSE_NAMES)
        results[name] = {
            "fit_seconds": train_dt,
            "train": train_eval,
            "val": val_eval,
        }

        print(f"  fit: {train_dt:.1f}s")
        print(f"  train acc: {train_eval['accuracy']:.4f}")
        print(f"  val   acc: {val_eval['accuracy']:.4f}")
        for cls, m in val_eval["report"].items():
            if cls in COARSE_NAMES:
                print(f"    {cls:10s} P={m['precision']:.3f} "
                      f"R={m['recall']:.3f} F1={m['f1-score']:.3f}")

        dump(pipe, out_dir / f"stage1_{name}.joblib")

    best_name = max(results, key=lambda k: results[k]["val"]["accuracy"])
    best_pipe = classifiers[best_name]
    print(f"\n>>> best stage1 on val: {best_name} "
          f"(coarse acc={results[best_name]['val']['accuracy']:.4f})")
    dump(best_pipe, out_dir / "best.joblib")  # legacy flat-model name

    # ---- Stage 2: per-family heads trained on GT routing ----
    if args.hierarchical:
        print("\n=== stage2: per-family heads ===")
        heads, head_results = train_stage2_heads(
            X_tr, y_fine_tr, X_va, y_fine_va, random_state=args.seed,
        )

        # ---- End-to-end evaluation on val with hard routing ----
        bundle = {
            "stage1": best_pipe,
            "stage2": heads,
            "fine_names": FINE_NAMES,
            "coarse_names": COARSE_NAMES,
            "fine_to_coarse": FINE_TO_COARSE,
            "coarse_to_fine": COARSE_TO_FINE,
        }
        fine_pred_va, coarse_pred_va = predict_fine_hard(bundle, X_va)
        e2e_acc = float((fine_pred_va == y_fine_va).mean())
        e2e_eval = evaluate_predictions(
            y_fine_va, fine_pred_va, FINE_NAMES,
        )
        print(f"\n>>> end-to-end fine val acc (hard routing): {e2e_acc:.4f}")
        for cls, m in e2e_eval["report"].items():
            if cls in FINE_NAMES:
                print(f"    {cls:12s} P={m['precision']:.3f} "
                      f"R={m['recall']:.3f} F1={m['f1-score']:.3f}")

        dump(bundle, out_dir / "hierarchical.joblib")
        print(f"saved: {out_dir / 'hierarchical.joblib'}")

        results["_stage2_heads"] = {
            COARSE_NAMES[k]: v for k, v in head_results.items()
        }
        results["_end_to_end_val"] = {
            "accuracy": e2e_acc,
            "report": e2e_eval["report"],
            "confusion": e2e_eval["confusion"],
            "labels": e2e_eval["labels"],
        }

    (out_dir / "results.json").write_text(json.dumps(
        {"best_stage1": best_name, "per_model": results,
         "hierarchical": bool(args.hierarchical)},
        indent=2, default=_json_default,
    ))
    print(f"saved: {out_dir / 'results.json'}")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         target_names: list[str]) -> dict:
    """Same shape as evaluate(), but takes pre-computed predictions."""
    acc = float((y_pred == y_true).mean())
    labels_present = sorted(
        set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist())
    )
    names_present = [target_names[i] for i in labels_present]
    report = classification_report(
        y_true, y_pred, labels=labels_present, target_names=names_present,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_present).tolist()
    return {"accuracy": acc, "report": report, "confusion": cm,
            "labels": labels_present}


def _json_default(o):
    """Make numpy arrays/ints JSON-serializable."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"not serializable: {type(o)}")


# ---------------------------------------------------------------------------
# Subcommand: predict
# ---------------------------------------------------------------------------

def cmd_predict(args: argparse.Namespace) -> None:
    feat_dir = scoped_feature_dir(args.features)
    print(f"Feature cache: {feat_dir}")
    test_X = np.load(feat_dir / "test_X.npy")
    test_paths = json.loads((feat_dir / "test_paths.json").read_text())
    print(f"test: {test_X.shape}")

    obj = load(args.model)
    is_hierarchical = isinstance(obj, dict) and "stage1" in obj
    print(f"loaded model: {args.model} "
          f"({'hierarchical bundle' if is_hierarchical else 'flat pipeline'})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if is_hierarchical:
        fine_pred, coarse_pred = predict_fine_hard(obj, test_X)
        # With hard routing, full 9-way probabilities aren't well-defined.
        # We emit stage-1 coarse probabilities + the within-family
        # probabilities from whichever head handled each sample.
        stage1: Pipeline = obj["stage1"]
        P_coarse = (stage1.predict_proba(test_X)
                    if hasattr(stage1, "predict_proba") else None)

        header = ["filename", "fine_label", "fine_name",
                  "coarse_label", "coarse_name"]
        if P_coarse is not None:
            header += [f"p_coarse_{c}" for c in COARSE_NAMES]
        header += ["p_within_family"]  # prob assigned to the picked fine class

        # Pre-compute within-family probability for the picked fine class
        within_prob = np.full(len(test_X), 1.0, dtype=np.float32)
        for coarse_idx, fine_members in obj["coarse_to_fine"].items():
            mask = coarse_pred == coarse_idx
            if not mask.any() or len(fine_members) < 2:
                continue
            if coarse_idx not in obj["stage2"]:
                # Head was skipped at train time; we already wrote a
                # fallback fine_pred in predict_fine_hard. No probability
                # to report — leave within_prob at 1.0 sentinel.
                continue
            head: Pipeline = obj["stage2"][coarse_idx]
            P_within = head.predict_proba(test_X[mask])  # (n, k)
            # Map fine_pred[mask] to column in head.classes_
            picked = fine_pred[mask]
            col_lookup = {int(c): i for i, c in enumerate(head.classes_)}
            cols = np.array([col_lookup[int(c)] for c in picked])
            within_prob[mask] = P_within[np.arange(len(picked)), cols]

        with out_path.open("w") as f:
            f.write(",".join(header) + "\n")
            for i, p in enumerate(test_paths):
                fname = Path(p).name
                fi, ci = int(fine_pred[i]), int(coarse_pred[i])
                row = [fname, str(fi), FINE_NAMES[fi],
                       str(ci), COARSE_NAMES[ci]]
                if P_coarse is not None:
                    row += [f"{P_coarse[i, k]:.6f}" for k in range(len(COARSE_NAMES))]
                row += [f"{within_prob[i]:.6f}"]
                f.write(",".join(row) + "\n")
    else:
        # Legacy flat pipeline path -- unchanged behaviour
        pipe: Pipeline = obj
        pred = pipe.predict(test_X)
        proba = pipe.predict_proba(test_X) if hasattr(pipe, "predict_proba") else None
        # Note: a legacy flat model trained pre-hierarchy might be either
        # coarse-only or 9-way; we just emit whatever it produces.
        n_classes = len(getattr(pipe, "classes_", FINE_NAMES))
        names = FINE_NAMES if n_classes == len(FINE_NAMES) else COARSE_NAMES

        header = ["filename", "label", "class_name"]
        if proba is not None:
            header += [f"p_{c}" for c in names[:proba.shape[1]]]

        with out_path.open("w") as f:
            f.write(",".join(header) + "\n")
            for i, p in enumerate(test_paths):
                fname = Path(p).name
                row = [fname, str(int(pred[i])), names[int(pred[i])]]
                if proba is not None:
                    row += [f"{proba[i, k]:.6f}" for k in range(proba.shape[1])]
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
    e.add_argument("--out", required=True,
                   help="Base feature cache dir; the active extractor kind "
                        "(from extractor_config.yaml) is appended automatically, "
                        "e.g. --out features -> features/spectral/")
    e.set_defaults(func=cmd_extract)

    t = sub.add_parser("train", help="Train all three stage-1 candidates, "
                                     "pick best on val, optionally train stage-2 heads")
    t.add_argument("--features", required=True,
                   help="Base feature cache dir; the active extractor kind is "
                        "appended automatically (must match the kind used at "
                        "extract time).")
    t.add_argument("--out", required=True, help="Where to write *.joblib + results.json")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--hierarchical", action="store_true",
                   help="Also train per-family stage-2 heads and save a "
                        "hierarchical bundle.")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="Predict on the test set")
    pr.add_argument("--features", required=True,
                    help="Base feature cache dir; the active extractor kind is "
                         "appended automatically.")
    pr.add_argument("--model", required=True,
                    help="Path to best.joblib or hierarchical.joblib "
                         "(auto-detected).")
    pr.add_argument("--out", required=True, help="Output CSV")
    pr.set_defaults(func=cmd_predict)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()