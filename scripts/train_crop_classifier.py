import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crop_classifier import IMAGENET_MEAN, IMAGENET_STD, create_classifier_model


class CropClassificationDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        class_to_index: dict[int, int],
        image_size: int,
        train: bool,
        crop_padding: float,
    ) -> None:
        self.samples = samples
        self.class_to_index = class_to_index
        self.crop_padding = crop_padding
        self.transform = self.build_transform(image_size, train)

    @staticmethod
    def build_transform(image_size: int, train: bool) -> transforms.Compose:
        normalize = transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
        if train:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=12),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, sample: dict) -> Image.Image:
        image_path = Path(sample["image_path"])
        with Image.open(image_path) as image_file:
            image_rgb = image_file.convert("RGB")
            if "bbox" not in sample:
                return image_rgb.copy()

            width, height = image_rgb.size
            x, y, w, h = [float(value) for value in sample["bbox"]]
            pad_w = w * self.crop_padding
            pad_h = h * self.crop_padding
            left = max(0, int(math.floor(x - pad_w)))
            top = max(0, int(math.floor(y - pad_h)))
            right = min(width, int(math.ceil(x + w + pad_w)))
            bottom = min(height, int(math.ceil(y + h + pad_h)))
            return image_rgb.crop((left, top, max(left + 1, right), max(top + 1, bottom)))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = self.load_image(sample)
        return self.transform(image), self.class_to_index[int(sample["category_id"])]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sampler(samples: list[dict]) -> WeightedRandomSampler:
    counts: dict[int, int] = {}
    for sample in samples:
        category_id = int(sample["category_id"])
        counts[category_id] = counts.get(category_id, 0) + 1
    weights = [1.0 / counts[int(sample["category_id"])] for sample in samples]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
            total_loss += float(loss.item()) * int(labels.numel())
    if total == 0:
        return 0.0, 0.0
    return correct / total, total_loss / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--architecture", default="tf_efficientnetv2_s.in21k_ft_in1k")
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--epochs", default=12, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--crop-padding", default=0.06, type=float)
    parser.add_argument("--output-dir", default=Path("weights"), type=Path)
    parser.add_argument("--weights-name", default="crop_classifier.pt")
    parser.add_argument("--config-name", default="crop_classifier.json")
    args = parser.parse_args()

    set_seed(args.seed)

    with args.manifest.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    class_ids = [int(class_id) for class_id in manifest["class_ids"]]
    class_to_index = {class_id: idx for idx, class_id in enumerate(class_ids)}
    train_samples = list(manifest["train_samples"])
    val_samples = list(manifest["val_samples"])

    train_dataset = CropClassificationDataset(train_samples, class_to_index, args.image_size, train=True, crop_padding=args.crop_padding)
    val_dataset = CropClassificationDataset(val_samples, class_to_index, args.image_size, train=False, crop_padding=args.crop_padding)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=build_sampler(train_samples),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_classifier_model(args.architecture, len(class_ids), pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_val_accuracy = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / args.weights_name
    config_path = args.output_dir / args.config_name

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            predictions = logits.argmax(dim=1)
            batch_size = int(labels.numel())
            running_total += batch_size
            running_correct += int((predictions == labels).sum().item())
            running_loss += float(loss.item()) * batch_size

        scheduler.step()

        train_accuracy = running_correct / max(1, running_total)
        train_loss = running_loss / max(1, running_total)
        val_accuracy, val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": args.architecture,
                    "image_size": args.image_size,
                    "class_ids": class_ids,
                    "best_val_accuracy": best_val_accuracy,
                },
                weights_path,
            )
            with config_path.open("w", encoding="utf-8") as config_file:
                json.dump(
                    {
                        "architecture": args.architecture,
                        "image_size": args.image_size,
                        "class_ids": class_ids,
                        "best_val_accuracy": best_val_accuracy,
                    },
                    config_file,
                    indent=2,
                )
            print(f"Saved best model to: {weights_path}")
            print(f"Saved config to: {config_path}")


if __name__ == "__main__":
    main()