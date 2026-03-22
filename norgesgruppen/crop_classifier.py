import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import timm
import torch


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def create_classifier_model(arch, num_classes, pretrained):
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def prepare_input(img_rgb, size):
    resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    t = resized.astype(np.float32) / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(t, (2, 0, 1)))


@dataclass
class ClassifierBundle:
    model: torch.nn.Module
    device: torch.device
    image_size: int
    class_ids: list[int]
    class_to_index: dict[int, int]

    @classmethod
    def load(cls, weights_path, config_path):
        if not weights_path.exists() or not config_path.exists():
            return None

        with open(config_path) as f:
            cfg = json.load(f)

        arch = str(cfg["architecture"])
        img_size = int(cfg["image_size"])
        class_ids = [int(c) for c in cfg["class_ids"]]
        model = create_classifier_model(arch, len(class_ids), pretrained=False)

        ckpt = torch.load(weights_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(dev).eval()

        return cls(
            model=model, device=dev, image_size=img_size,
            class_ids=class_ids,
            class_to_index={cid: i for i, cid in enumerate(class_ids)},
        )

    def predict(self, img_rgb):
        x = prepare_input(img_rgb, self.image_size).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)
        best = int(np.argmax(probs))
        return self.class_ids[best], float(probs[best]), probs
