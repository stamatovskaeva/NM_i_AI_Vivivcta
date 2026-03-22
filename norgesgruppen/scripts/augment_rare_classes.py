"""
Offline augmentation pass for underrepresented classes.

For every class that has fewer than `--target` training samples we generate
augmented crops and save them as real .jpg files so they can be added to the
training manifest without re-reading the (sometimes very large) shelf images on
every epoch.

Augmentation intensity scales with how rare a class is:
  - any class below target:        flips + colour jitter
  - below half target:             + rotation ±20° + scale jitter
  - below quarter target:          + perspective warp  (simulates scrunched /
                                     squeezed packaging, rotated facings, etc.)
  - below 10 samples:              all of the above, run multiple passes

Usage
-----
python scripts/augment_rare_classes.py \
    --manifest artifacts/crop_classifier_manifest.json \
    --output-dir artifacts/augmented_crops \
    --target 60 \
    --seed 42

The script updates the manifest in-place (keeps originals, appends augmented
samples tagged `"source": "augmented"`).  Pass --dry-run to only print stats.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
except ImportError:
    sys.exit("Install Pillow and numpy first: pip install Pillow numpy")


# ---------------------------------------------------------------------------
# Augmentation primitives
# ---------------------------------------------------------------------------

def _random_flip(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def _color_jitter(
    img: Image.Image,
    rng: random.Random,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    hue_shift: float = 0.06,
) -> Image.Image:
    # Brightness
    factor = 1.0 + rng.uniform(-brightness, brightness)
    img = ImageEnhance.Brightness(img).enhance(max(0.1, factor))
    # Contrast
    factor = 1.0 + rng.uniform(-contrast, contrast)
    img = ImageEnhance.Contrast(img).enhance(max(0.1, factor))
    # Saturation
    factor = 1.0 + rng.uniform(-saturation, saturation)
    img = ImageEnhance.Color(img).enhance(max(0.0, factor))
    # Sharpness / blur alternation
    if rng.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
    elif rng.random() < 0.15:
        img = ImageEnhance.Sharpness(img).enhance(rng.uniform(1.5, 3.0))
    return img


def _rotate(img: Image.Image, rng: random.Random, max_degrees: float = 20.0) -> Image.Image:
    angle = rng.uniform(-max_degrees, max_degrees)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=None)


def _scale_jitter(img: Image.Image, rng: random.Random, scale_range: Tuple[float, float] = (0.80, 1.15)) -> Image.Image:
    """Zoom in/out and re-crop to original size."""
    w, h = img.size
    scale = rng.uniform(*scale_range)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    # Centre-crop or pad back to original size
    if scale >= 1.0:
        x0 = (new_w - w) // 2
        y0 = (new_h - h) // 2
        resized = resized.crop((x0, y0, x0 + w, y0 + h))
    else:
        canvas = Image.new("RGB", (w, h), (128, 128, 128))
        x0 = (w - new_w) // 2
        y0 = (h - new_h) // 2
        canvas.paste(resized, (x0, y0))
        resized = canvas
    return resized


def _perspective_warp(img: Image.Image, rng: random.Random, distortion: float = 0.18) -> Image.Image:
    """
    Random perspective transform – simulates a product that is slightly
    crumpled, at an angle, or viewed off-axis (like a scrunched coffee bag).
    """
    w, h = img.size
    d = distortion
    # perturb the four corners independently
    def jitter(val: float, size: int) -> float:
        return val + rng.uniform(-d * size, d * size)

    coeffs = _find_coeffs(
        # destination corners (original rectangle)
        [(0, 0), (w, 0), (w, h), (0, h)],
        # source corners (perturbed)
        [
            (jitter(0, w), jitter(0, h)),
            (jitter(w, w), jitter(0, h)),
            (jitter(w, w), jitter(h, h)),
            (jitter(0, w), jitter(h, h)),
        ],
    )
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def _find_coeffs(pa: List[Tuple[float, float]], pb: List[Tuple[float, float]]):
    """Compute perspective transform coefficients (8-element list for PIL)."""
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    import numpy as np
    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)
    res = np.linalg.solve(A, B)
    return np.array(res).flatten().tolist()


def augment_image(
    img: Image.Image,
    rng: random.Random,
    level: int,  # 1=mild, 2=medium, 3=strong, 4=very strong
) -> Image.Image:
    """Apply augmentations according to severity level."""
    img = _random_flip(img, rng)
    img = _color_jitter(img, rng)

    if level >= 2:
        img = _rotate(img, rng, max_degrees=20.0)
        img = _scale_jitter(img, rng)

    if level >= 3:
        img = _perspective_warp(img, rng, distortion=0.15)

    if level >= 4:
        # Extra pass: flip again + heavier colour jitter
        img = _random_flip(img, rng)
        img = _color_jitter(img, rng, brightness=0.55, contrast=0.55, saturation=0.55)
        img = _perspective_warp(img, rng, distortion=0.22)

    return img


# ---------------------------------------------------------------------------
# Crop extraction
# ---------------------------------------------------------------------------

def load_crop(sample: dict, min_size: int = 16) -> Image.Image | None:
    path = Path(sample["image_path"])
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None

    if "bbox" in sample:
        x, y, bw, bh = [int(round(v)) for v in sample["bbox"]]
        iw, ih = img.size
        # Tiny padding (3 %)
        pad_x = max(1, int(bw * 0.03))
        pad_y = max(1, int(bh * 0.03))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(iw, x + bw + pad_x)
        y2 = min(ih, y + bh + pad_y)
        crop = img.crop((x1, y1, x2, y2))
    else:
        crop = img  # reference images are already single-product

    cw, ch = crop.size
    if cw < min_size or ch < min_size:
        return None
    return crop


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline augmentation for rare classes")
    parser.add_argument("--manifest",    default="artifacts/crop_classifier_manifest.json", type=Path)
    parser.add_argument("--output-dir",  default="artifacts/augmented_crops", type=Path)
    parser.add_argument("--target",      default=60, type=int,
                        help="Target number of train samples per class. "
                             "Classes already at or above this are skipped.")
    parser.add_argument("--seed",        default=42, type=int)
    parser.add_argument("--quality",     default=92, type=int,
                        help="JPEG save quality (0-95)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print stats only, do not write files or edit manifest")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading manifest: {args.manifest}")
    with args.manifest.open() as fh:
        manifest = json.load(fh)

    train_samples: list[dict] = manifest["train_samples"]

    # ---- group train samples by class, skip already-augmented ones so we
    #      don't double-augment on re-runs
    from collections import defaultdict
    by_class: dict[int, list[dict]] = defaultdict(list)
    for s in train_samples:
        if s.get("source") != "augmented":
            by_class[int(s["category_id"])].append(s)

    # Counts including any previously saved augmented samples
    existing_aug: dict[int, int] = defaultdict(int)
    for s in train_samples:
        if s.get("source") == "augmented":
            existing_aug[int(s["category_id"])] += 1

    target = args.target
    new_samples: list[dict] = []

    # Determine augmentation levels based on rarity
    #   level 1 : < target
    #   level 2 : < target // 2
    #   level 3 : < target // 4
    #   level 4 : < 10

    stats = {"skipped_enough": 0, "skipped_no_images": 0, "generated": 0, "classes_augmented": 0}

    classes_needing_aug = {
        cid: samples
        for cid, samples in by_class.items()
        if (len(samples) + existing_aug[cid]) < target
    }

    print(f"Classes below target ({target}): {len(classes_needing_aug)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cid, original_samples in sorted(classes_needing_aug.items()):
        current_count = len(original_samples) + existing_aug[cid]
        need = target - current_count

        # Pick augmentation level
        if current_count < 10:
            level = 4
        elif current_count < target // 4:
            level = 3
        elif current_count < target // 2:
            level = 2
        else:
            level = 1

        class_dir = args.output_dir / str(cid)
        if not args.dry_run:
            class_dir.mkdir(parents=True, exist_ok=True)

        # We repeat originals in a round-robin to avoid always augmenting the
        # same source image
        source_pool = list(original_samples)
        if not source_pool:
            stats["skipped_no_images"] += 1
            continue

        rng.shuffle(source_pool)
        generated = 0

        for i in range(need):
            src = source_pool[i % len(source_pool)]
            crop = load_crop(src)
            if crop is None:
                continue

            aug = augment_image(crop, rng, level)

            if not args.dry_run:
                out_name = f"aug_{existing_aug[cid] + generated:04d}.jpg"
                out_path = class_dir / out_name
                aug.save(out_path, "JPEG", quality=args.quality)
                new_samples.append({
                    "source": "augmented",
                    "image_path": str(out_path.resolve()),
                    "category_id": cid,
                    "bbox": None,   # already cropped
                })

            generated += 1

        stats["generated"]          += generated
        stats["classes_augmented"]  += 1

        if generated:
            final_count = current_count + generated
            print(
                f"  class {cid:3d}: had {current_count:3d} + generated {generated:3d} "
                f"= {final_count:3d} (target {target}, level {level})"
            )

    print()
    print("=== Summary ===")
    print(f"  Classes augmented  : {stats['classes_augmented']}")
    print(f"  New crops generated: {stats['generated']}")

    if args.dry_run:
        print("(dry-run – nothing saved)")
        return

    if not new_samples:
        print("Nothing new to add to manifest.")
        return

    # Append augmented samples and save manifest
    manifest["train_samples"] = train_samples + new_samples
    with args.manifest.open("w") as fh:
        json.dump(manifest, fh)

    print(f"  Updated manifest   : {args.manifest}")
    print(f"  Total train samples: {len(manifest['train_samples'])}")


if __name__ == "__main__":
    main()
