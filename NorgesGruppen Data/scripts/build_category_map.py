import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", default=Path("artifacts/category_map.json"), type=Path)
    args = parser.parse_args()

    with args.annotations.open("r", encoding="utf-8") as annotations_file:
        coco = json.load(annotations_file)

    categories = sorted(coco["categories"], key=lambda item: int(item["id"]))
    category_to_name = {str(item["id"]): item["name"] for item in categories}
    name_to_category = {item["name"]: int(item["id"]) for item in categories}

    payload = {
        "num_categories": len(categories),
        "category_to_name": category_to_name,
        "name_to_category": name_to_category,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)

    print(f"Saved category map to: {args.output}")
    print(f"Total categories: {len(categories)}")


if __name__ == "__main__":
    main()