import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
)


def register_ultralytics_safe_globals() -> None:
    torch.serialization.add_safe_globals(
        [DetectionModel, SegmentationModel, ClassificationModel, PoseModel, OBBModel]
    )


def patch_torch_load_for_ultralytics() -> None:
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8m.pt", help="Base model or checkpoint")
    parser.add_argument("--imgsz", default=1280, type=int)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch", default=4, type=int)
    parser.add_argument("--project", default="artifacts/train", type=Path)
    parser.add_argument("--name", default="yolov8m_ngd")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--device", default=None, help="Training device (e.g., cpu, 0, 0,1). Defaults to auto")
    parser.add_argument(
        "--aug-profile",
        default="balanced",
        choices=["light", "balanced", "aggressive"],
        help="Augmentation strength profile",
    )
    args = parser.parse_args()

    args.project.mkdir(parents=True, exist_ok=True)

    register_ultralytics_safe_globals()
    patch_torch_load_for_ultralytics()

    train_device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")

    aug_profiles = {
        "light": {
            "degrees": 6.0,
            "shear": 2.0,
            "perspective": 0.0002,
            "flipud": 0.0,
            "fliplr": 0.5,
            "scale": 0.35,
            "translate": 0.08,
            "hsv_h": 0.012,
            "hsv_s": 0.45,
            "hsv_v": 0.30,
            "mosaic": 0.5,
            "copy_paste": 0.0,
            "mixup": 0.0,
            "close_mosaic": 10,
        },
        "balanced": {
            "degrees": 10.0,
            "shear": 3.0,
            "perspective": 0.0003,
            "flipud": 0.01,
            "fliplr": 0.5,
            "scale": 0.45,
            "translate": 0.10,
            "hsv_h": 0.015,
            "hsv_s": 0.60,
            "hsv_v": 0.35,
            "mosaic": 0.8,
            "copy_paste": 0.05,
            "mixup": 0.05,
            "close_mosaic": 12,
        },
        "aggressive": {
            "degrees": 12.0,
            "shear": 4.0,
            "perspective": 0.0004,
            "flipud": 0.02,
            "fliplr": 0.5,
            "scale": 0.6,
            "translate": 0.12,
            "hsv_h": 0.020,
            "hsv_s": 0.80,
            "hsv_v": 0.45,
            "mosaic": 1.0,
            "copy_paste": 0.25,
            "mixup": 0.10,
            "close_mosaic": 15,
        },
    }
    aug = aug_profiles[args.aug_profile]

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        device=train_device,
        pretrained=True,
        cos_lr=True,
        optimizer="AdamW",
        degrees=aug["degrees"],
        shear=aug["shear"],
        perspective=aug["perspective"],
        flipud=aug["flipud"],
        fliplr=aug["fliplr"],
        scale=aug["scale"],
        translate=aug["translate"],
        hsv_h=aug["hsv_h"],
        hsv_s=aug["hsv_s"],
        hsv_v=aug["hsv_v"],
        mosaic=aug["mosaic"],
        copy_paste=aug["copy_paste"],
        mixup=aug["mixup"],
        label_smoothing=0.05,
        close_mosaic=aug["close_mosaic"],
    )


if __name__ == "__main__":
    main()