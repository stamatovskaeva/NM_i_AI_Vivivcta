import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
)

from crop_classifier import ClassifierBundle
from reference_features import ReferenceIndex, compute_descriptor, crop_from_bbox


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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


def parse_image_id(image_path: Path) -> int:
    return int(image_path.stem.split("_")[-1])


def list_images(input_dir: Path) -> list[Path]:
    return [
        image_path
        for image_path in sorted(input_dir.iterdir())
        if image_path.suffix.lower() in IMAGE_SUFFIXES
    ]


def load_class_remap(class_map_path: Path | None) -> dict[int, int]:
    if class_map_path is None or not class_map_path.exists():
        return {}

    with class_map_path.open("r", encoding="utf-8") as class_map_file:
        raw_map = json.load(class_map_file)

    if isinstance(raw_map, dict) and "model_to_competition" in raw_map:
        raw_map = raw_map["model_to_competition"]

    return {int(key): int(value) for key, value in raw_map.items()}


def remap_category_id(category_id: int, class_remap: dict[int, int]) -> int:
    if not class_remap:
        return category_id
    return class_remap.get(category_id, category_id)


def maybe_rerank_category(
    image_rgb: np.ndarray,
    bbox: list[float],
    category_id: int,
    score: float,
    reference_index: ReferenceIndex | None,
    min_crop_size: int,
    min_similarity: float,
    switch_margin: float,
) -> int:
    if reference_index is None:
        return category_id

    crop = crop_from_bbox(image_rgb, bbox)
    if crop is None:
        return category_id

    crop_height, crop_width = crop.shape[:2]
    if min(crop_height, crop_width) < min_crop_size:
        return category_id

    descriptor = compute_descriptor(crop)
    best_category_id, best_similarity = reference_index.best_match(descriptor)
    detector_similarity = reference_index.score_for_category(descriptor, category_id)

    if best_category_id == category_id:
        return category_id
    if best_similarity < min_similarity:
        return category_id
    if detector_similarity is None:
        if score < 0.4 and best_similarity >= min_similarity + switch_margin:
            return best_category_id
        return category_id
    if score < 0.85 and best_similarity - detector_similarity >= switch_margin:
        return best_category_id
    return category_id


def maybe_rerank_with_classifier(
    image_rgb: np.ndarray,
    bbox: list[float],
    category_id: int,
    score: float,
    classifier_bundle: ClassifierBundle | None,
    min_crop_size: int,
    min_detector_score: float,
    max_detector_score: float,
    min_classifier_score: float,
    min_margin: float,
) -> int:
    if classifier_bundle is None:
        return category_id
    if score < min_detector_score or score > max_detector_score:
        return category_id

    crop = crop_from_bbox(image_rgb, bbox)
    if crop is None:
        return category_id
    if min(crop.shape[:2]) < min_crop_size:
        return category_id

    predicted_category_id, predicted_score, probabilities = classifier_bundle.predict(crop)
    if predicted_category_id == category_id:
        return category_id

    detector_index = classifier_bundle.class_to_index.get(int(category_id))
    detector_probability = float(probabilities[detector_index]) if detector_index is not None else 0.0

    if predicted_score >= min_classifier_score and predicted_score - detector_probability >= min_margin:
        return predicted_category_id
    return category_id


def infer_with_ultralytics_pt(
    images: list[Path],
    model_path: Path,
    conf: float,
    iou: float,
    class_remap: dict[int, int],
    classifier_bundle: ClassifierBundle | None,
    classifier_min_detector_score: float,
    classifier_max_detector_score: float,
    classifier_min_prob: float,
    classifier_margin: float,
    reference_index: ReferenceIndex | None,
    min_crop_size: int,
    min_similarity: float,
    switch_margin: float,
) -> list[dict]:
    from ultralytics import YOLO

    register_ultralytics_safe_globals()
    patch_torch_load_for_ultralytics()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    predictions: list[dict] = []

    for image_path in images:
        image_id = parse_image_id(image_path)
        results = model.predict(
            source=str(image_path),
            device=device,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            image_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
            for idx in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
                detector_category_id = remap_category_id(int(boxes.cls[idx].item()), class_remap)
                bbox = [
                    round(x1, 1),
                    round(y1, 1),
                    round(x2 - x1, 1),
                    round(y2 - y1, 1),
                ]
                score = round(float(boxes.conf[idx].item()), 3)
                category_id = maybe_rerank_with_classifier(
                    image_rgb=image_rgb,
                    bbox=bbox,
                    category_id=detector_category_id,
                    score=score,
                    classifier_bundle=classifier_bundle,
                    min_crop_size=min_crop_size,
                    min_detector_score=classifier_min_detector_score,
                    max_detector_score=classifier_max_detector_score,
                    min_classifier_score=classifier_min_prob,
                    min_margin=classifier_margin,
                )
                category_id = maybe_rerank_category(
                    image_rgb=image_rgb,
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    reference_index=reference_index,
                    min_crop_size=min_crop_size,
                    min_similarity=min_similarity,
                    switch_margin=switch_margin,
                )
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": score,
                    }
                )
    return predictions


def preprocess_for_onnx(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    import cv2

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return tensor, (original_w, original_h)


def infer_with_onnx(
    images: list[Path],
    model_path: Path,
    conf: float,
    size: int,
    class_remap: dict[int, int],
    classifier_bundle: ClassifierBundle | None,
    classifier_min_detector_score: float,
    classifier_max_detector_score: float,
    classifier_min_prob: float,
    classifier_margin: float,
    reference_index: ReferenceIndex | None,
    min_crop_size: int,
    min_similarity: float,
    switch_margin: float,
) -> list[dict]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    predictions: list[dict] = []

    for image_path in images:
        image_id = parse_image_id(image_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        tensor, (original_w, original_h) = preprocess_for_onnx(image_bgr, size)
        outputs = session.run(None, {input_name: tensor})
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        output = outputs[0]
        if output.ndim == 3:
            output = output[0]

        for row in output:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, score, cls_id = row[:6]
            score = float(score)
            if score < conf:
                continue

            sx = original_w / size
            sy = original_h / size
            x1, x2 = float(x1) * sx, float(x2) * sx
            y1, y2 = float(y1) * sy, float(y2) * sy
            detector_category_id = remap_category_id(int(cls_id), class_remap)
            bbox = [
                round(x1, 1),
                round(y1, 1),
                round(max(0.0, x2 - x1), 1),
                round(max(0.0, y2 - y1), 1),
            ]
            category_id = maybe_rerank_with_classifier(
                image_rgb=image_rgb,
                bbox=bbox,
                category_id=detector_category_id,
                score=score,
                classifier_bundle=classifier_bundle,
                min_crop_size=min_crop_size,
                min_detector_score=classifier_min_detector_score,
                max_detector_score=classifier_max_detector_score,
                min_classifier_score=classifier_min_prob,
                min_margin=classifier_margin,
            )
            category_id = maybe_rerank_category(
                image_rgb=image_rgb,
                bbox=bbox,
                category_id=category_id,
                score=score,
                reference_index=reference_index,
                min_crop_size=min_crop_size,
                min_similarity=min_similarity,
                switch_margin=switch_margin,
            )

            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": round(score, 3),
                }
            )

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", default="weights/best.pt", type=Path)
    parser.add_argument("--conf", default=0.15, type=float)
    parser.add_argument("--iou", default=0.6, type=float)
    parser.add_argument("--imgsz", default=1280, type=int)
    parser.add_argument("--class-map", default=None, type=Path)
    parser.add_argument("--classifier-weights", default=Path("weights/crop_classifier.pt"), type=Path)
    parser.add_argument("--classifier-config", default=Path("weights/crop_classifier.json"), type=Path)
    parser.add_argument("--classifier-min-detector-score", default=0.25, type=float)
    parser.add_argument("--classifier-max-detector-score", default=0.80, type=float)
    parser.add_argument("--classifier-min-prob", default=0.62, type=float)
    parser.add_argument("--classifier-margin", default=0.25, type=float)
    parser.add_argument("--reference-features", default=Path("weights/reference_features.npy"), type=Path)
    parser.add_argument("--reference-manifest", default=Path("weights/reference_manifest.json"), type=Path)
    parser.add_argument("--ref-min-crop-size", default=56, type=int)
    parser.add_argument("--ref-min-similarity", default=0.40, type=float)
    parser.add_argument("--ref-switch-margin", default=0.10, type=float)
    args = parser.parse_args()

    images = list_images(args.input)
    if not images:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output_file:
            json.dump([], output_file)
        return

    class_remap = load_class_remap(args.class_map)
    classifier_bundle = ClassifierBundle.load(args.classifier_weights, args.classifier_config)
    reference_index = ReferenceIndex.load(args.reference_features, args.reference_manifest)

    if args.weights.suffix == ".onnx":
        predictions = infer_with_onnx(
            images,
            args.weights,
            args.conf,
            args.imgsz,
            class_remap,
            classifier_bundle,
            args.classifier_min_detector_score,
            args.classifier_max_detector_score,
            args.classifier_min_prob,
            args.classifier_margin,
            reference_index,
            args.ref_min_crop_size,
            args.ref_min_similarity,
            args.ref_switch_margin,
        )
    else:
        predictions = infer_with_ultralytics_pt(
            images,
            args.weights,
            args.conf,
            args.iou,
            class_remap,
            classifier_bundle,
            args.classifier_min_detector_score,
            args.classifier_max_detector_score,
            args.classifier_min_prob,
            args.classifier_margin,
            reference_index,
            args.ref_min_crop_size,
            args.ref_min_similarity,
            args.ref_switch_margin,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(predictions, output_file)


if __name__ == "__main__":
    with torch.no_grad():
        main()