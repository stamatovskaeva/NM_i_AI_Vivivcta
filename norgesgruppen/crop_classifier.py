import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import timm
import torch


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def create_classifier_model(architecture: str, num_classes: int, pretrained: bool) -> torch.nn.Module:
    return timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)


def prepare_classifier_input(image_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    tensor = np.transpose(tensor, (2, 0, 1))
    return torch.from_numpy(tensor)


@dataclass
class ClassifierBundle:
    model: torch.nn.Module
    device: torch.device
    image_size: int
    class_ids: list[int]
    class_to_index: dict[int, int]

    @classmethod
    def load(cls, weights_path: Path, config_path: Path) -> "ClassifierBundle | None":
        if not weights_path.exists() or not config_path.exists():
            return None

        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        architecture = str(config["architecture"])
        image_size = int(config["image_size"])
        class_ids = [int(class_id) for class_id in config["class_ids"]]
        model = create_classifier_model(architecture, len(class_ids), pretrained=False)

        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        return cls(
            model=model,
            device=device,
            image_size=image_size,
            class_ids=class_ids,
            class_to_index={class_id: idx for idx, class_id in enumerate(class_ids)},
        )

    def predict(self, image_rgb: np.ndarray) -> tuple[int, float, np.ndarray]:
        input_tensor = prepare_classifier_input(image_rgb, self.image_size).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)
        best_index = int(np.argmax(probabilities))
        return self.class_ids[best_index], float(probabilities[best_index]), probabilities
