import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def copy_group(files: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        shutil.copy2(file_path, destination / file_path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=Path("data/crops"), type=Path)
    parser.add_argument("--train", default=Path("data/crops_train"), type=Path)
    parser.add_argument("--val", default=Path("data/crops_val"), type=Path)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    args.train.mkdir(parents=True, exist_ok=True)
    args.val.mkdir(parents=True, exist_ok=True)

    copied_train = 0
    copied_val = 0

    for class_dir in sorted(path for path in args.input.iterdir() if path.is_dir()):
        files = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            continue

        grouped: dict[str, list[Path]] = defaultdict(list)
        for file_path in files:
            prefix = file_path.stem.split("-")[0]
            grouped[prefix].append(file_path)

        keys = list(grouped.keys())
        random.shuffle(keys)

        split_index = max(1, int(len(keys) * args.val_ratio)) if len(keys) > 1 else 0
        val_keys = set(keys[:split_index])

        train_files: list[Path] = []
        val_files: list[Path] = []
        for key, grouped_files in grouped.items():
            if key in val_keys:
                val_files.extend(grouped_files)
            else:
                train_files.extend(grouped_files)

        copy_group(train_files, args.train / class_dir.name)
        copy_group(val_files, args.val / class_dir.name)
        copied_train += len(train_files)
        copied_val += len(val_files)

    print(f"Train crops copied: {copied_train}")
    print(f"Val crops copied: {copied_val}")


if __name__ == "__main__":
    main()