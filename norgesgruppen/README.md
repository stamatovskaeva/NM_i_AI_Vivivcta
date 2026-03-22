# norgesgruppen

Full task folder for the NorgesGruppen Data object-detection challenge.

## What is inside this folder

### Runtime / submission files

- `run.py`
- `crop_classifier.py`
- `reference_features.py`
- `weights/best.pt`
- `weights/reference_features.npy`
- `weights/reference_manifest.json`

### Training and utility scripts

- `scripts/` (data prep, training helpers, validation scripts)

### Base model checkpoints used during training

- `models/yolov8m.pt`
- `models/yolov8n.pt`

### Environment files

- `requirements.txt`
- `requirements-train.txt`
- `requirements-local-macos.txt`

## Competition run command

The sandbox runs:

```bash
python run.py --input /data/images --output /output/predictions.json
```

## Output contract

`run.py` writes a JSON array of detections:

```json
[
  {
    "image_id": 42,
    "category_id": 0,
    "bbox": [120.5, 45.0, 80.0, 110.0],
    "score": 0.923
  }
]
```

- `image_id`: integer parsed from filename `img_XXXXX.jpg`
- `category_id`: competition class id
- `bbox`: COCO format `[x, y, w, h]`
- `score`: confidence in `[0, 1]`

## Build submission zip

From inside this folder:

```bash
zip -r ../submission.zip run.py crop_classifier.py reference_features.py weights -x "*.DS_Store" "__MACOSX/*"
```

Do **not** include markdown/text docs in the competition zip. The scoring environment is pre-provisioned and does not install dependencies from `requirements*.txt`.
