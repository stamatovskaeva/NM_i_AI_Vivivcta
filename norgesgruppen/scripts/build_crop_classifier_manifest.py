import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def normalize_product_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def load_overrides(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as override_file:
        payload = json.load(override_file)
    return {normalize_product_name(key): int(value) for key, value in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--product-root", default=None, type=Path)
    parser.add_argument("--metadata", default=None, type=Path)
    parser.add_argument("--output", default=Path("artifacts/crop_classifier_manifest.json"), type=Path)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--min-size", default=24, type=int)
    parser.add_argument("--include-reference-in-val", action="store_true")
    parser.add_argument("--reference-val-ratio", default=0.0, type=float)
    parser.add_argument("--overrides", default=None, type=Path)
    args = parser.parse_args()

    random.seed(args.seed)

    with args.annotations.open("r", encoding="utf-8") as annotations_file:
        coco = json.load(annotations_file)

    images_by_id = {int(image["id"]): image for image in coco["images"]}
    image_ids = sorted(images_by_id.keys())
    random.shuffle(image_ids)
    split_index = int(len(image_ids) * args.val_ratio)
    val_image_ids = set(image_ids[:split_index])

    categories = sorted(coco["categories"], key=lambda item: int(item["id"]))
    category_names = {int(category["id"]): str(category["name"]) for category in categories}

    train_samples: list[dict] = []
    val_samples: list[dict] = []

    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        image = images_by_id.get(image_id)
        if image is None:
            continue
        width = float(annotation["bbox"][2])
        height = float(annotation["bbox"][3])
        if min(width, height) < args.min_size:
            continue

        sample = {
            "source": "shelf",
            "image_path": str((args.images / image["file_name"]).resolve()),
            "category_id": int(annotation["category_id"]),
            "bbox": [float(value) for value in annotation["bbox"]],
            "image_id": image_id,
        }

        if image_id in val_image_ids:
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    unmatched_reference_products: list[str] = []
    if args.product_root is not None:
        metadata_path = args.metadata or (args.product_root / "metadata.json")
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        category_by_name = {normalize_product_name(name): category_id for category_id, name in category_names.items()}
        overrides = load_overrides(args.overrides)

        reference_samples: list[dict] = []
        for product in metadata["products"]:
            if not product.get("has_images"):
                continue
            product_name = str(product["product_name"])
            normalized_name = normalize_product_name(product_name)
            category_id = overrides.get(normalized_name, category_by_name.get(normalized_name))
            if category_id is None:
                unmatched_reference_products.append(product_name)
                continue

            product_code = str(product["product_code"])
            product_dir = args.product_root / product_code
            if not product_dir.exists():
                continue

            product_images = [path for path in sorted(product_dir.iterdir()) if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if not product_images:
                continue

            val_count = 0
            if args.include_reference_in_val and args.reference_val_ratio > 0:
                val_count = max(1, int(len(product_images) * args.reference_val_ratio))
            val_paths = set(random.sample(product_images, k=min(val_count, len(product_images)))) if val_count > 0 else set()

            for image_path in product_images:
                sample = {
                    "source": "reference",
                    "image_path": str(image_path.resolve()),
                    "category_id": int(category_id),
                    "product_code": product_code,
                    "product_name": product_name,
                    "image_type": image_path.stem,
                }
                if image_path in val_paths:
                    val_samples.append(sample)
                else:
                    train_samples.append(sample)
                reference_samples.append(sample)

    class_ids = sorted({int(sample["category_id"]) for sample in train_samples + val_samples})

    def count_by_class(samples: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample in samples:
            key = str(int(sample["category_id"]))
            counts[key] = counts.get(key, 0) + 1
        return counts

    payload = {
        "version": 1,
        "seed": args.seed,
        "class_ids": class_ids,
        "categories": [{"id": category_id, "name": category_names[category_id]} for category_id in class_ids],
        "train_samples": train_samples,
        "val_samples": val_samples,
        "train_class_counts": count_by_class(train_samples),
        "val_class_counts": count_by_class(val_samples),
        "unmatched_reference_products": unmatched_reference_products,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)

    print(f"Saved manifest to: {args.output}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Classes: {len(class_ids)}")
    print(f"Unmatched reference products: {len(unmatched_reference_products)}")


if __name__ == "__main__":
    main()