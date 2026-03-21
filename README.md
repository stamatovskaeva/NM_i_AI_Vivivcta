# NM_i_AI_Vivivcta

Starter workspace for the NorgesGruppen Data object-detection competition.

## 1) What to put in this repository first

Add these folders locally (not all of them go into the final submission zip):

- `data/` (your downloaded training dataset, local only)
- `weights/` (trained model files)
- `artifacts/` (predictions, logs, temporary files)

This repo already includes:

- `run.py` (required submission entrypoint)
- `scripts/train_yolov8.py` (local training helper)
- `scripts/build_crop_classifier_manifest.py` (build crop-classifier manifest from shelf crops + reference images)
- `scripts/train_crop_classifier.py` (train a timm/PyTorch crop classifier)
- `scripts/quick_validate.py` (local run.py smoke test)
- `scripts/crop_from_coco.py` (crop products from COCO bboxes, old-pipeline style)
- `scripts/split_crops.py` (grouped train/val crop split by product prefix)
- `scripts/coco_to_yolo.py` (COCO -> YOLO dataset conversion)
- `scripts/build_category_map.py` (persist category mappings artifact)
- `requirements-train.txt` (version pins matching sandbox)

## 2) Local setup

Use Python 3.11 locally to match sandbox package versions.

macOS (no conda needed):

```bash
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-local-macos.txt
```

Linux / sandbox-matching installs:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-train.txt
```

Why install failed earlier: your previous `.venv` used Python 3.14, and macOS cannot install `onnxruntime-gpu` wheels.

## 3) Prepare data for YOLOv8

Download and unzip the COCO dataset from the competition website.

Recommended structure:

```text
data/
	coco/
		images/
			img_00001.jpg
			...
		annotations.json
	yolo/
		images/
			train/
			val/
		labels/
			train/
			val/
		data.yaml
```

Convert to YOLO format:

```bash
python scripts/coco_to_yolo.py \
	--annotations data/coco/annotations.json \
	--images data/coco/images \
	--output data/yolo
```

Build reusable category map artifact (helps keep training/inference mapping deterministic):

```bash
python scripts/build_category_map.py \
	--annotations data/coco/annotations.json \
	--output artifacts/category_map.json
```

Build compact product-reference descriptors for inference-time reranking:

```bash
python scripts/build_reference_index.py \
	--annotations train/annotations.json \
	--product-root NM_NGD_product_images \
	--output-features weights/reference_features.npy \
	--output-manifest weights/reference_manifest.json
```

This maps product reference images to competition category IDs by normalized product name. In the current data, 328 of 329 products map automatically; the remaining mismatch is reported in the manifest so you can add an override later if needed.

Build a crop-classifier manifest for a second-stage classifier:

```bash
python scripts/build_crop_classifier_manifest.py \
	--annotations train/annotations.json \
	--images train/images \
	--product-root NM_NGD_product_images \
	--output artifacts/crop_classifier_manifest.json
```

This uses an image-level train/val split for shelf crops to reduce leakage, and adds reference images into training by default.

Optional: recreate your previous crop-based workflow for classifier experiments:

```bash
python scripts/crop_from_coco.py \
	--annotations data/coco/annotations.json \
	--images data/coco/images \
	--output data/crops

python scripts/split_crops.py \
	--input data/crops \
	--train data/crops_train \
	--val data/crops_val
```

## 4) Train a first strong baseline

```bash
python scripts/train_yolov8.py \
	--data data/yolo/data.yaml \
	--model yolov8m.pt \
	--imgsz 1280 \
	--epochs 80 \
	--batch 4 \
	--aug-profile balanced \
	--project artifacts/train
```

Then copy your best weights to:

- `weights/best.pt` (for direct Ultralytics inference), or
- `weights/model.onnx` (for ONNX inference)

Optional but recommended for better product identification:

- `weights/crop_classifier.pt`
- `weights/crop_classifier.json`
- `weights/reference_features.npy`
- `weights/reference_manifest.json`

Train a second-stage crop classifier:

```bash
python scripts/train_crop_classifier.py \
	--manifest artifacts/crop_classifier_manifest.json \
	--architecture tf_efficientnetv2_s.in21k_ft_in1k \
	--image-size 224 \
	--epochs 12 \
	--batch-size 64 \
	--output-dir weights
```

Training uses weighted sampling rather than brute-force class duplication.

For quick smoke validation on CPU before long runs, first create a reduced manifest:

```bash
python scripts/make_smoke_manifest.py \
	--input artifacts/crop_classifier_manifest.json \
	--output artifacts/crop_classifier_manifest_smoke.json \
	--train-max 3000 \
	--val-max 600

python scripts/train_crop_classifier.py \
	--manifest artifacts/crop_classifier_manifest_smoke.json \
	--architecture efficientnet_b0 \
	--image-size 224 \
	--epochs 1 \
	--batch-size 64 \
	--num-workers 0 \
	--output-dir artifacts/smoke_classifier
```

## 5) Local inference test (same contract as competition)

```bash
python run.py --input data/coco/images --output artifacts/predictions.json
python scripts/quick_validate.py --predictions artifacts/predictions.json
```

If your model class ids differ from competition ids, pass a remap file:

```bash
python run.py --input data/coco/images --output artifacts/predictions.json --class-map artifacts/class_map.json
```

If `weights/reference_features.npy` and `weights/reference_manifest.json` exist, `run.py` will use them to conservatively rerank low-confidence detections based on the product reference images. You can tune this behavior with:

```bash
python run.py \
	--input train/images \
	--output artifacts/predictions.json \
	--ref-min-crop-size 56 \
	--ref-min-similarity 0.35 \
	--ref-switch-margin 0.08
```

If `weights/crop_classifier.pt` and `weights/crop_classifier.json` exist, `run.py` will first use the crop classifier on medium-confidence detections before the reference-feature reranker. This is the recommended hybrid path:

```bash
python run.py \
	--input train/images \
	--output artifacts/predictions.json \
	--classifier-min-detector-score 0.25 \
	--classifier-max-detector-score 0.80 \
	--classifier-min-prob 0.62 \
	--classifier-margin 0.25
```

## 6) Create submission zip

Your zip root must include `run.py`.

Example (from repo root):

```bash
zip -r submission.zip run.py reference_features.py crop_classifier.py weights -x "*.DS_Store" "__MACOSX/*"
```

Keep under limits (420MB total weights, max 10 Python files).

## 7) Practical strategy (first week)

1. Submit once with current `run.py` + pretrained weights to validate pipeline.
2. Fine-tune with `nc=356` to align category IDs from `train/annotations.json`.
3. Increase image size and use TTA at inference if timeout allows.
4. Improve product-ID accuracy with product image data through the crop classifier and reference-image reranker.
5. Save one conservative submission for final selection.
