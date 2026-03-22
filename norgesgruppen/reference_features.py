import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_product_name(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return " ".join(s.split())


def trim_border(img, ratio=0.08):
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        return img
    dy, dx = int(round(h * ratio)), int(round(w * ratio))
    y1 = min(max(dy, 0), max(h - 2, 0))
    y2 = max(y1 + 1, h - dy)
    x1 = min(max(dx, 0), max(w - 2, 0))
    x2 = max(x1 + 1, w - dx)
    return img[y1:y2, x1:x2]


def build_fg_mask(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask = np.logical_or(sat > 30, val < 235)
    if float(mask.mean()) < 0.05:
        mask = np.ones(mask.shape, dtype=bool)
    return mask.astype(np.uint8) * 255


def compute_descriptor(img_rgb):
    trimmed = trim_border(img_rgb)
    hsv = cv2.cvtColor(trimmed, cv2.COLOR_RGB2HSV)
    mask = build_fg_mask(trimmed)

    hs_hist = cv2.calcHist([hsv], [0, 1], mask, [24, 8], [0, 180, 0, 256]).astype(np.float32).ravel()
    v_hist = cv2.calcHist([hsv], [2], mask, [8], [0, 256]).astype(np.float32).ravel()
    if hs_hist.sum() > 0:
        hs_hist /= hs_hist.sum()
    if v_hist.sum() > 0:
        v_hist /= v_hist.sum()

    # spatial colour features on a 3x3 grid
    grid = 3
    h, w = hsv.shape[:2]
    grid_feats = []
    for r in range(grid):
        for c in range(grid):
            y1, y2 = (r * h) // grid, ((r + 1) * h) // grid
            x1, x2 = (c * w) // grid, ((c + 1) * w) // grid
            cell = hsv[y1:y2, x1:x2]
            m = mask[y1:y2, x1:x2] > 0
            if cell.size == 0 or not np.any(m):
                grid_feats.extend((0.0, 0.0, 0.0))
                continue
            px = cell[m]
            grid_feats.extend((
                float(px[:, 0].mean()) / 180.0,
                float(px[:, 1].mean()) / 255.0,
                float(px[:, 2].mean()) / 255.0,
            ))

    feat = np.concatenate((hs_hist, v_hist, np.asarray(grid_feats, dtype=np.float32))).astype(np.float32)
    norm = float(np.linalg.norm(feat))
    if norm > 0:
        feat /= norm
    return feat


@dataclass
class ReferenceIndex:
    features: np.ndarray
    category_ids: np.ndarray
    product_codes: list[str]
    product_names: list[str]

    @classmethod
    def load(cls, feat_path, manifest_path):
        if not feat_path.exists() or not manifest_path.exists():
            return None

        features = np.load(feat_path).astype(np.float32)
        with open(manifest_path) as f:
            manifest = json.load(f)

        entries = manifest.get("entries", [])
        cat_ids = np.asarray([int(e["category_id"]) for e in entries], dtype=np.int32)
        codes = [str(e["product_code"]) for e in entries]
        names = [str(e["product_name"]) for e in entries]

        if features.ndim != 2 or len(entries) != features.shape[0]:
            raise ValueError("Reference feature files are inconsistent")

        return cls(features=features, category_ids=cat_ids,
                   product_codes=codes, product_names=names)

    def best_match(self, desc):
        sims = self.features @ desc
        idx = int(np.argmax(sims))
        return int(self.category_ids[idx]), float(sims[idx])

    def score_for_category(self, desc, cat_id):
        mask = self.category_ids == int(cat_id)
        if not np.any(mask):
            return None
        return float(np.max(self.features[mask] @ desc))


def crop_from_bbox(img_rgb, bbox):
    h, w = img_rgb.shape[:2]
    x, y, bw, bh = bbox
    x1 = max(0, min(w - 1, int(np.floor(x))))
    y1 = max(0, min(h - 1, int(np.floor(y))))
    x2 = max(x1 + 1, min(w, int(np.ceil(x + bw))))
    y2 = max(y1 + 1, min(h, int(np.ceil(y + bh))))
    if x2 <= x1 or y2 <= y1:
        return None
    return img_rgb[y1:y2, x1:x2]
