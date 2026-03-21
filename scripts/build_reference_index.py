import argparse
import json
import re
import unicodedata
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


def load_override_map(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as override_file:
        payload = json.load(override_file)
    return {normalize_product_name(key): int(value) for key, value in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--product-root", required=True, type=Path)
    parser.add_argument("--metadata", default=None, type=Path)
    parser.add_argument("--output-features", default=Path("weights/reference_features.npy"), type=Path)
    parser.add_argument("--output-manifest", default=Path("weights/reference_manifest.json"), type=Path)
    parser.add_argument("--overrides", default=None, type=Path)
    args = parser.parse_args()

    metadata_path = args.metadata or (args.product_root / "metadata.json")

    with args.annotations.open("r", encoding="utf-8") as annotations_file:
        coco = json.load(annotations_file)
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    category_by_name = {
        normalize_product_name(category["name"]): int(category["id"])
        for category in coco["categories"]
    }
    overrides = load_override_map(args.overrides)

    feature_rows: list[np.ndarray] = []
    manifest_entries: list[dict] = []
    unmatched: list[str] = []

    for product in metadata["products"]:
        if not product.get("has_images"):
            continue

        product_name = str(product["product_name"])
        normalized_name = normalize_product_name(product_name)
        category_id = overrides.get(normalized_name, category_by_name.get(normalized_name))
        if category_id is None:
            unmatched.append(product_name)
            continue

        product_code = str(product["product_code"])
        product_dir = args.product_root / product_code
        if not product_dir.exists():
            continue

        for image_path in sorted(path for path in product_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            feature_rows.append(compute_descriptor(image_rgb))
            manifest_entries.append(
                {
                    "category_id": int(category_id),
                    "product_code": product_code,
                    "product_name": product_name,
                    "image_name": image_path.name,
                }
            )

    if not feature_rows:
        raise ValueError("No reference features were generated")

    args.output_features.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    feature_matrix = np.stack(feature_rows).astype(np.float32)
    np.save(args.output_features, feature_matrix)

    payload = {
        "version": 1,
        "num_entries": len(manifest_entries),
        "num_categories": len({entry["category_id"] for entry in manifest_entries}),
        "unmatched_products": unmatched,
        "entries": manifest_entries,
    }
    with args.output_manifest.open("w", encoding="utf-8") as manifest_file:
        json.dump(payload, manifest_file, indent=2, ensure_ascii=False)

    print(f"Saved features to: {args.output_features}")
    print(f"Saved manifest to: {args.output_manifest}")
    print(f"Reference entries: {len(manifest_entries)}")
    print(f"Matched categories: {payload['num_categories']}")
    print(f"Unmatched products: {len(unmatched)}")
    if unmatched:
        print("First unmatched product:", unmatched[0])


if __name__ == "__main__":
    main()