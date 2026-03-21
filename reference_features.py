import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def normalize_product_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def trim_border(image_rgb: np.ndarray, trim_ratio: float = 0.08) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    if height < 8 or width < 8:
        return image_rgb

    y_margin = int(round(height * trim_ratio))
    x_margin = int(round(width * trim_ratio))

    y1 = min(max(y_margin, 0), max(height - 2, 0))
    y2 = max(y1 + 1, height - y_margin)
    x1 = min(max(x_margin, 0), max(width - 2, 0))
    x2 = max(x1 + 1, width - x_margin)
    return image_rgb[y1:y2, x1:x2]


def build_foreground_mask(image_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = np.logical_or(saturation > 30, value < 235)
    if float(mask.mean()) < 0.05:
        mask = np.ones(mask.shape, dtype=bool)
    return mask.astype(np.uint8) * 255


def compute_descriptor(image_rgb: np.ndarray) -> np.ndarray:
    trimmed = trim_border(image_rgb)
    hsv = cv2.cvtColor(trimmed, cv2.COLOR_RGB2HSV)
    mask = build_foreground_mask(trimmed)

    hs_hist = cv2.calcHist([hsv], [0, 1], mask, [24, 8], [0, 180, 0, 256]).astype(np.float32)
    v_hist = cv2.calcHist([hsv], [2], mask, [8], [0, 256]).astype(np.float32)

    hs_hist = hs_hist.reshape(-1)
    v_hist = v_hist.reshape(-1)
    if float(hs_hist.sum()) > 0:
        hs_hist /= float(hs_hist.sum())
    if float(v_hist.sum()) > 0:
        v_hist /= float(v_hist.sum())

    grid_features: list[float] = []
    grid_size = 3
    height, width = hsv.shape[:2]
    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            y1 = (row_idx * height) // grid_size
            y2 = ((row_idx + 1) * height) // grid_size
            x1 = (col_idx * width) // grid_size
            x2 = ((col_idx + 1) * width) // grid_size

            cell = hsv[y1:y2, x1:x2]
            cell_mask = mask[y1:y2, x1:x2] > 0
            if cell.size == 0 or not np.any(cell_mask):
                grid_features.extend((0.0, 0.0, 0.0))
                continue

            masked_pixels = cell[cell_mask]
            grid_features.extend(
                (
                    float(masked_pixels[:, 0].mean()) / 180.0,
                    float(masked_pixels[:, 1].mean()) / 255.0,
                    float(masked_pixels[:, 2].mean()) / 255.0,
                )
            )

    feature = np.concatenate((hs_hist, v_hist, np.asarray(grid_features, dtype=np.float32))).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    if norm > 0:
        feature /= norm
    return feature


@dataclass
class ReferenceIndex:
    features: np.ndarray
    category_ids: np.ndarray
    product_codes: list[str]
    product_names: list[str]

    @classmethod
    def load(cls, features_path: Path, manifest_path: Path) -> "ReferenceIndex | None":
        if not features_path.exists() or not manifest_path.exists():
            return None

        features = np.load(features_path).astype(np.float32)
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)

        entries = manifest.get("entries", [])
        category_ids = np.asarray([int(entry["category_id"]) for entry in entries], dtype=np.int32)
        product_codes = [str(entry["product_code"]) for entry in entries]
        product_names = [str(entry["product_name"]) for entry in entries]

        if features.ndim != 2 or len(entries) != features.shape[0]:
            raise ValueError("Reference feature files are inconsistent")

        return cls(
            features=features,
            category_ids=category_ids,
            product_codes=product_codes,
            product_names=product_names,
        )

    def best_match(self, descriptor: np.ndarray) -> tuple[int, float]:
        similarities = self.features @ descriptor
        best_idx = int(np.argmax(similarities))
        return int(self.category_ids[best_idx]), float(similarities[best_idx])

    def score_for_category(self, descriptor: np.ndarray, category_id: int) -> float | None:
        matches = self.category_ids == int(category_id)
        if not np.any(matches):
            return None
        similarities = self.features[matches] @ descriptor
        return float(np.max(similarities))


def crop_from_bbox(image_rgb: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    height, width = image_rgb.shape[:2]
    x, y, w, h = bbox
    x1 = max(0, min(width - 1, int(np.floor(x))))
    y1 = max(0, min(height - 1, int(np.floor(y))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x + w))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y + h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return image_rgb[y1:y2, x1:x2]
