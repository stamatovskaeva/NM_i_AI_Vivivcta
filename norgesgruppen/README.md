# norgesgruppen

Minimal runtime package for NorgesGruppen Data submission.

## What this folder contains

- `run.py` — submission entrypoint (required)
- `crop_classifier.py` — optional second-stage crop classifier helper
- `reference_features.py` — reference-based reranking helper
- `weights/best.pt` — trained detector weights (primary model)
- `weights/reference_features.npy` + `weights/reference_manifest.json` — optional reference reranking assets
- `requirements.txt` — local reproducibility only (not used by competition runner)

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

Do **not** include markdown/text docs in the competition zip. The scoring environment is pre-provisioned and does not install dependencies from `requirements.txt`.
