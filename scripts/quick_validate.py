import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args()

    with args.predictions.open("r", encoding="utf-8") as predictions_file:
        predictions = json.load(predictions_file)

    if not isinstance(predictions, list):
        raise ValueError("predictions.json must contain a JSON array")

    required_keys = {"image_id", "category_id", "bbox", "score"}
    for index, item in enumerate(predictions[:2000]):
        if not isinstance(item, dict):
            raise ValueError(f"Prediction at index {index} is not an object")
        missing = required_keys.difference(item.keys())
        if missing:
            raise ValueError(f"Prediction at index {index} missing keys: {sorted(missing)}")

        bbox = item["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Prediction at index {index} has invalid bbox")
        if item["score"] < 0 or item["score"] > 1:
            raise ValueError(f"Prediction at index {index} has score outside [0,1]")

    print(f"OK: {len(predictions)} predictions validated")


if __name__ == "__main__":
    main()