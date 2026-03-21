import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reduced manifest for quick smoke training")
    parser.add_argument("--input", default=Path("artifacts/crop_classifier_manifest.json"), type=Path)
    parser.add_argument("--output", default=Path("artifacts/crop_classifier_manifest_smoke.json"), type=Path)
    parser.add_argument("--train-max", default=3000, type=int)
    parser.add_argument("--val-max", default=600, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)

    with args.input.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_samples = list(manifest.get("train_samples", []))
    val_samples = list(manifest.get("val_samples", []))

    if len(train_samples) > args.train_max:
        train_samples = random.sample(train_samples, args.train_max)
    if len(val_samples) > args.val_max:
        val_samples = random.sample(val_samples, args.val_max)

    class_ids = sorted({int(s["category_id"]) for s in train_samples + val_samples})

    out_manifest = {
        **manifest,
        "class_ids": class_ids,
        "train_samples": train_samples,
        "val_samples": val_samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out_manifest, f)

    print(f"Saved smoke manifest: {args.output}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Classes present: {len(class_ids)}")


if __name__ == "__main__":
    main()
