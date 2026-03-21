import json
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def status(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def warn_status(condition_is_clean: bool) -> str:
    return "OK" if condition_is_clean else "WARN"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ann_path = ROOT / "train" / "annotations.json"
    yolo_yaml_path = ROOT / "data" / "yolo" / "data.yaml"
    ref_manifest_path = ROOT / "weights" / "reference_manifest.json"
    crop_manifest_path = ROOT / "artifacts" / "crop_classifier_manifest.json"
    smoke_pred_path = ROOT / "artifacts" / "predictions_smoke.json"

    print("=== NM Preflight Audit ===")

    if not ann_path.exists():
        print(f"[{status(False)}] Missing annotations: {ann_path}")
        return

    ann = load_json(ann_path)
    category_ids = sorted(int(c["id"]) for c in ann["categories"])
    max_id = category_ids[-1]
    inferred_nc = max_id + 1

    contiguous = category_ids == list(range(inferred_nc))
    print(f"[{status(True)}] annotations categories: {len(category_ids)} (id range {category_ids[0]}..{max_id})")
    print(f"[{status(contiguous)}] category ids contiguous: {contiguous}")

    blank_names = [int(c["id"]) for c in ann["categories"] if not str(c.get("name", "")).strip()]
    print(f"[{warn_status(len(blank_names) == 0)}] blank category names: {blank_names if blank_names else 'none'}")

    if yolo_yaml_path.exists():
        yolo_text = yolo_yaml_path.read_text(encoding="utf-8")
        nc_line = next((line for line in yolo_text.splitlines() if line.startswith("nc:")), None)
        yolo_nc = int(nc_line.split(":", 1)[1].strip()) if nc_line else None
        print(f"[{status(yolo_nc == inferred_nc)}] yolo nc: {yolo_nc} (expected {inferred_nc})")

        names_text = yolo_text.split("names:", 1)[1].strip() if "names:" in yolo_text else "[]"
        try:
            names = ast.literal_eval(names_text)
            print(f"[{status(len(names) == inferred_nc)}] yolo names length: {len(names)}")
        except Exception:
            print(f"[{status(False)}] could not parse names list in {yolo_yaml_path}")
    else:
        print(f"[{status(False)}] missing yolo yaml: {yolo_yaml_path}")

    if ref_manifest_path.exists():
        ref = load_json(ref_manifest_path)
        print(f"[{status(True)}] reference entries: {ref.get('num_entries', 0)}")
        print(f"[{status(True)}] reference categories matched: {ref.get('num_categories', 0)}")
        print(f"[{status(True)}] unmatched reference products: {len(ref.get('unmatched_products', []))}")
    else:
        print(f"[{status(False)}] missing reference manifest: {ref_manifest_path}")

    if crop_manifest_path.exists():
        crop = load_json(crop_manifest_path)
        class_ids = [int(x) for x in crop.get("class_ids", [])]
        train_samples = crop.get("train_samples", [])
        val_samples = crop.get("val_samples", [])
        print(f"[{status(True)}] crop manifest classes: {len(class_ids)}")
        print(f"[{status(True)}] crop train/val samples: {len(train_samples)}/{len(val_samples)}")

        missing_in_crop = sorted(set(category_ids) - set(class_ids))
        print(f"[{status(len(missing_in_crop) == 0)}] classes missing from crop manifest: {missing_in_crop if missing_in_crop else 'none'}")
    else:
        print(f"[{status(False)}] missing crop manifest: {crop_manifest_path}")

    print(f"[{status(smoke_pred_path.exists())}] smoke predictions file exists: {smoke_pred_path.exists()}")

    weights_dir = ROOT / "weights"
    weight_exts = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
    weight_files = [p for p in weights_dir.glob("**/*") if p.is_file() and p.suffix.lower() in weight_exts]
    print(f"[{status(len(weight_files) <= 3)}] weight files in weights/: {len(weight_files)} (submission limit is 3)")
    for wf in sorted(weight_files):
        print(f"  - {wf.relative_to(ROOT)} ({wf.stat().st_size / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    main()
