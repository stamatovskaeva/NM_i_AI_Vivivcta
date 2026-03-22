# NM_i_AI_Vivivcta

This repository contains multiple NM/AI tasks organized by folder.

## Task folders

- `norgesgruppen/` — NorgesGruppen Data object detection task (runtime, training scripts, weights, and requirements)

## Working rule

Task-specific files should live inside their task folder, not at repository root.

For the NorgesGruppen task, use `norgesgruppen/README.md` as the source of truth.
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
