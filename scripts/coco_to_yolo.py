import argparse
import json
import random
import shutil
from pathlib import Path


def yolo_line(category_id: int, bbox: list[float], width: int, height: int) -> str:
    x, y, w, h = bbox
    xc = (x + w / 2.0) / width
    yc = (y + h / 2.0) / height
    nw = w / width
    nh = h / height
    return f"{category_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/yolo"), type=Path)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--num-classes",
        default=None,
        type=int,
        help="Optional override for number of classes. Defaults to max(category_id)+1 from annotations.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    with args.annotations.open("r", encoding="utf-8") as annotations_file:
        coco = json.load(annotations_file)

    output_root = args.output.resolve()
    output_images_train = output_root / "images" / "train"
    output_images_val = output_root / "images" / "val"
    output_labels_train = output_root / "labels" / "train"
    output_labels_val = output_root / "labels" / "val"

    for directory in [
        output_images_train,
        output_images_val,
        output_labels_train,
        output_labels_val,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    images = list(coco["images"])
    random.shuffle(images)
    split_idx = int(len(images) * args.val_ratio)
    val_ids = {int(image["id"]) for image in images[:split_idx]}

    for image in images:
        image_id = int(image["id"])
        image_name = image["file_name"]
        width = int(image["width"])
        height = int(image["height"])

        source_image = args.images / image_name
        if not source_image.exists():
            continue

        is_val = image_id in val_ids
        out_image = (output_images_val if is_val else output_images_train) / image_name
        out_label = (output_labels_val if is_val else output_labels_train) / f"{Path(image_name).stem}.txt"

        shutil.copy2(source_image, out_image)

        lines: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            category_id = int(annotation["category_id"])
            lines.append(yolo_line(category_id, annotation["bbox"], width, height))

        out_label.write_text("\n".join(lines), encoding="utf-8")

    category_ids = [int(category["id"]) for category in coco["categories"]]
    inferred_num_classes = max(category_ids) + 1 if category_ids else 0
    num_classes = inferred_num_classes if args.num_classes is None else int(args.num_classes)
    if num_classes < inferred_num_classes:
        raise ValueError(
            f"--num-classes ({num_classes}) is smaller than max category id + 1 ({inferred_num_classes})."
        )

    names = [f"class_{idx}" for idx in range(num_classes)]
    for category in coco["categories"]:
        category_id = int(category["id"])
        if 0 <= category_id < num_classes:
            category_name = str(category.get("name", "")).strip()
            names[category_id] = category_name if category_name else f"class_{category_id}"

    data_yaml = (
        f"path: {output_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {num_classes}\n"
        f"names: {names}\n"
    )
    (output_root / "data.yaml").write_text(data_yaml, encoding="utf-8")

    print(f"Prepared YOLO dataset at: {output_root}")
    print(f"Train images: {len(images) - split_idx}")
    print(f"Val images: {split_idx}")
    print(f"Classes (nc): {num_classes}")


if __name__ == "__main__":
    main()