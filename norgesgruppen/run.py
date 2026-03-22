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


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _register_safe_globals():
    torch.serialization.add_safe_globals(
        [DetectionModel, SegmentationModel, ClassificationModel, PoseModel, OBBModel]
    )


def _patch_torch_load():
    _orig = torch.load

    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _patched


def parse_image_id(p):
    return int(p.stem.split("_")[-1])


def list_images(input_dir):
    return [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]


def load_class_remap(path):
    if path is None or not path.exists():
        return {}
    with path.open() as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "model_to_competition" in raw:
        raw = raw["model_to_competition"]
    return {int(k): int(v) for k, v in raw.items()}


def remap_cat(cat_id, remap):
    if not remap:
        return cat_id
    return remap.get(cat_id, cat_id)


def rerank_by_ref(img_rgb, bbox, cat_id, score, ref_index,
                  min_crop_px, min_sim, margin):
    """Try to correct category using reference-image similarity."""
    if ref_index is None:
        return cat_id

    crop = crop_from_bbox(img_rgb, bbox)
    if crop is None:
        return cat_id
    if min(crop.shape[:2]) < min_crop_px:
        return cat_id

    desc = compute_descriptor(crop)
    best_cat, best_sim = ref_index.best_match(desc)
    det_sim = ref_index.score_for_category(desc, cat_id)

    if best_cat == cat_id:
        return cat_id
    if best_sim < min_sim:
        return cat_id
    if det_sim is None:
        if score < 0.4 and best_sim >= min_sim + margin:
            return best_cat
        return cat_id
    if score < 0.85 and best_sim - det_sim >= margin:
        return best_cat
    return cat_id


def rerank_by_classifier(img_rgb, bbox, cat_id, score, clf,
                         min_crop_px, min_det, max_det, min_cls, margin):
    """Try to correct category using the crop classifier."""
    if clf is None:
        return cat_id
    if score < min_det or score > max_det:
        return cat_id

    crop = crop_from_bbox(img_rgb, bbox)
    if crop is None:
        return cat_id
    if min(crop.shape[:2]) < min_crop_px:
        return cat_id

    pred_cat, pred_score, probs = clf.predict(crop)
    if pred_cat == cat_id:
        return cat_id

    det_idx = clf.class_to_index.get(int(cat_id))
    det_prob = float(probs[det_idx]) if det_idx is not None else 0.0

    if pred_score >= min_cls and pred_score - det_prob >= margin:
        return pred_cat
    return cat_id


def infer_with_ultralytics_pt(
    images, model_path, conf, iou, class_remap,
    clf_bundle, clf_min_det, clf_max_det, clf_min_prob, clf_margin,
    ref_index, min_crop_px, min_sim, switch_margin,
):
    from ultralytics import YOLO

    _register_safe_globals()
    _patch_torch_load()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    preds = []

    for img_path in images:
        image_id = parse_image_id(img_path)
        results = model.predict(str(img_path), device=device, conf=conf, iou=iou, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            img_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                det_cat = remap_cat(int(boxes.cls[i].item()), class_remap)
                bbox = [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]
                score = round(float(boxes.conf[i].item()), 3)

                cat_id = rerank_by_classifier(
                    img_rgb, bbox, det_cat, score, clf_bundle,
                    min_crop_px, clf_min_det, clf_max_det, clf_min_prob, clf_margin,
                )
                cat_id = rerank_by_ref(
                    img_rgb, bbox, cat_id, score, ref_index,
                    min_crop_px, min_sim, switch_margin,
                )
                preds.append({
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": bbox,
                    "score": score,
                })
    return preds


def _prep_onnx(img_bgr, size):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    t = resized.astype(np.float32) / 255.0
    t = np.transpose(t, (2, 0, 1))[None, ...]
    return t, (w, h)


def infer_with_onnx(
    images, model_path, conf, size, class_remap,
    clf_bundle, clf_min_det, clf_max_det, clf_min_prob, clf_margin,
    ref_index, min_crop_px, min_sim, switch_margin,
):
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    preds = []

    for img_path in images:
        image_id = parse_image_id(img_path)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        tensor, (orig_w, orig_h) = _prep_onnx(img_bgr, size)
        outputs = sess.run(None, {input_name: tensor})
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        out = outputs[0]
        if out.ndim == 3:
            out = out[0]

        for row in out:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, sc, cid = row[:6]
            sc = float(sc)
            if sc < conf:
                continue

            sx, sy = orig_w / size, orig_h / size
            x1, x2 = float(x1) * sx, float(x2) * sx
            y1, y2 = float(y1) * sy, float(y2) * sy
            det_cat = remap_cat(int(cid), class_remap)
            bbox = [round(x1, 1), round(y1, 1),
                    round(max(0.0, x2 - x1), 1), round(max(0.0, y2 - y1), 1)]

            cat_id = rerank_by_classifier(
                img_rgb, bbox, det_cat, sc, clf_bundle,
                min_crop_px, clf_min_det, clf_max_det, clf_min_prob, clf_margin,
            )
            cat_id = rerank_by_ref(
                img_rgb, bbox, cat_id, sc, ref_index,
                min_crop_px, min_sim, switch_margin,
            )
            preds.append({
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": bbox,
                "score": round(sc, 3),
            })

    return preds


def main():
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
    parser.add_argument("--clf-min-det", default=0.25, type=float)
    parser.add_argument("--clf-max-det", default=0.80, type=float)
    parser.add_argument("--clf-min-prob", default=0.62, type=float)
    parser.add_argument("--clf-margin", default=0.25, type=float)
    parser.add_argument("--reference-features", default=Path("weights/reference_features.npy"), type=Path)
    parser.add_argument("--reference-manifest", default=Path("weights/reference_manifest.json"), type=Path)
    parser.add_argument("--ref-min-crop-size", default=56, type=int)
    parser.add_argument("--ref-min-similarity", default=0.40, type=float)
    parser.add_argument("--ref-switch-margin", default=0.10, type=float)
    args = parser.parse_args()

    images = list_images(args.input)
    if not images:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([], f)
        return

    class_remap = load_class_remap(args.class_map)
    clf = ClassifierBundle.load(args.classifier_weights, args.classifier_config)
    ref_index = ReferenceIndex.load(args.reference_features, args.reference_manifest)

    shared = dict(
        class_remap=class_remap,
        clf_bundle=clf,
        clf_min_det=args.clf_min_det,
        clf_max_det=args.clf_max_det,
        clf_min_prob=args.clf_min_prob,
        clf_margin=args.clf_margin,
        ref_index=ref_index,
        min_crop_px=args.ref_min_crop_size,
        min_sim=args.ref_min_similarity,
        switch_margin=args.ref_switch_margin,
    )

    if args.weights.suffix == ".onnx":
        preds = infer_with_onnx(images, args.weights, args.conf, args.imgsz, **shared)
    else:
        preds = infer_with_ultralytics_pt(images, args.weights, args.conf, args.iou, **shared)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(preds, f)


if __name__ == "__main__":
    with torch.no_grad():
        main()