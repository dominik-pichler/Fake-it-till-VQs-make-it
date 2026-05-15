from pathlib import Path
from typing import List
import joblib
import numpy as np
from PIL import Image


def load_model(model_path: Path):
    return joblib.load(model_path)


def preprocess_image(img_path: Path, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()  # example: sklearn model expecting 1D features


def classify_images(
    img_paths: List[Path],
    model_path: Path,
) -> List[int]:
    model = load_model(model_path)
    features = np.stack([preprocess_image(p) for p in img_paths])
    preds = model.predict(features)
    return preds.astype(int).tolist()