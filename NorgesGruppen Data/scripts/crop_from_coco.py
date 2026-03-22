import argparse
import json
from pathlib import Path

from PIL import Image


def sanitize_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/crops"), type=Path)
    parser.add_argument("--min-size", default=16, type=int)
    args = parser.parse_args()

    with args.annotations.open("r", encoding="utf-8") as annotations_file:
        coco = json.load(annotations_file)

    image_by_id = {image["id"]: image for image in coco["images"]}
    category_name = {category["id"]: category["name"] for category in coco["categories"]}

    args.output.mkdir(parents=True, exist_ok=True)

    counts: dict[int, int] = {}
    saved = 0
    skipped = 0

    for annotation in coco["annotations"]:
        image = image_by_id.get(annotation["image_id"])
        if image is None:
            skipped += 1
            continue

        image_path = args.images / image["file_name"]
        if not image_path.exists():
            skipped += 1
            continue

        category_id = int(annotation["category_id"])
        label_dir = args.output / f"{category_id:03d}_{sanitize_name(category_name.get(category_id, 'unknown'))}"
        label_dir.mkdir(parents=True, exist_ok=True)

        x, y, w, h = annotation["bbox"]
        left = max(0, int(round(x)))
        top = max(0, int(round(y)))
        right = max(left + 1, int(round(x + w)))
        bottom = max(top + 1, int(round(y + h)))

        if right - left < args.min_size or bottom - top < args.min_size:
            skipped += 1
            continue

        with Image.open(image_path) as image_file:
            image_rgb = image_file.convert("RGB")
            crop = image_rgb.crop((left, top, right, bottom))

        idx = counts.get(category_id, 0)
        counts[category_id] = idx + 1

        product_code = str(annotation.get("product_code", "unknown"))
        out_name = f"{product_code}-{image['id']:05d}-{annotation['id']:06d}-{idx:05d}.jpg"
        crop.save(label_dir / out_name, quality=95)
        saved += 1

    print(f"Saved crops: {saved}")
    print(f"Skipped annotations: {skipped}")
    print(f"Classes with crops: {len(counts)}")


if __name__ == "__main__":
    main()