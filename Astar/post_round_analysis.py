from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import config
from api_client import AstarClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]


def _normalise_probs(tensor: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    clipped = np.clip(tensor.astype(np.float64), eps, 1.0)
    sums = clipped.sum(axis=2, keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    return clipped / sums


def _seed_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    pred_n = _normalise_probs(pred)
    truth_n = _normalise_probs(truth)

    ce = -np.sum(truth_n * np.log(pred_n), axis=2)
    kl = np.sum(truth_n * (np.log(truth_n) - np.log(pred_n)), axis=2)
    acc = (np.argmax(pred_n, axis=2) == np.argmax(truth_n, axis=2)).astype(np.float64)

    pred_mass = pred_n.mean(axis=(0, 1))
    truth_mass = truth_n.mean(axis=(0, 1))
    class_bias = pred_mass - truth_mass

    dominant_pred = np.argmax(pred_n, axis=2)
    dominant_truth = np.argmax(truth_n, axis=2)
    mismatch_rate = float(np.mean(dominant_pred != dominant_truth))

    return {
        "mean_kl": float(np.mean(kl)),
        "mean_cross_entropy": float(np.mean(ce)),
        "top1_accuracy": float(np.mean(acc)),
        "mismatch_rate": mismatch_rate,
        "predicted_class_mass": pred_mass.tolist(),
        "truth_class_mass": truth_mass.tolist(),
        "class_mass_bias": class_bias.tolist(),
    }


def _overall_metrics(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_rows:
        return {}

    mean_kl = float(np.mean([row["mean_kl"] for row in seed_rows]))
    mean_ce = float(np.mean([row["mean_cross_entropy"] for row in seed_rows]))
    mean_acc = float(np.mean([row["top1_accuracy"] for row in seed_rows]))
    mean_mismatch = float(np.mean([row["mismatch_rate"] for row in seed_rows]))

    class_bias = np.mean(np.array([row["class_mass_bias"] for row in seed_rows]), axis=0)

    return {
        "mean_kl": mean_kl,
        "mean_cross_entropy": mean_ce,
        "top1_accuracy": mean_acc,
        "mismatch_rate": mean_mismatch,
        "mean_class_mass_bias": class_bias.tolist(),
        "class_names": CLASS_NAMES,
    }


def _select_round(my_rounds: list[dict[str, Any]], round_id: str | None) -> dict[str, Any]:
    if round_id:
        row = next((r for r in my_rounds if r.get("id") == round_id), None)
        if row is None:
            raise RuntimeError(f"Round id {round_id} was not found in /my-rounds.")
        return row

    candidates = [
        r for r in my_rounds
        if r.get("status") in {"scoring", "completed"}
    ]
    if not candidates:
        raise RuntimeError("No round in scoring/completed state available from /my-rounds.")

    candidates.sort(key=lambda r: r.get("round_number", 0), reverse=True)
    return candidates[0]


def _round_slug(round_row: dict[str, Any]) -> str:
    number = round_row.get("round_number")
    round_id = round_row.get("id", "unknown")
    if number is None:
        return f"round_{round_id}"
    return f"round_{int(number)}_{round_id}"


def run_post_round_analysis(round_id: str | None, output_dir: Path) -> Path:
    client = AstarClient()
    my_rounds = client.get_my_rounds()
    round_row = _select_round(my_rounds, round_id)

    selected_round_id = str(round_row["id"])
    round_label = _round_slug(round_row)
    out_dir = output_dir / round_label
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Analyzing round %s (round_number=%s, status=%s)",
        selected_round_id,
        round_row.get("round_number"),
        round_row.get("status"),
    )

    seed_rows: list[dict[str, Any]] = []
    seed_raw: dict[str, Any] = {}

    for seed_idx in range(config.NUM_SEEDS):
        payload = client.get_analysis(selected_round_id, seed_idx)
        pred = np.array(payload["prediction"], dtype=np.float64)
        truth = np.array(payload["ground_truth"], dtype=np.float64)

        metrics = _seed_metrics(pred, truth)
        metrics["seed_index"] = seed_idx
        metrics["api_score"] = payload.get("score")

        seed_rows.append(metrics)
        seed_raw[str(seed_idx)] = {
            "score": payload.get("score"),
            "width": payload.get("width"),
            "height": payload.get("height"),
            "initial_grid": payload.get("initial_grid"),
            "prediction": payload.get("prediction"),
            "ground_truth": payload.get("ground_truth"),
        }

        logger.info(
            "Seed %d: score=%s KL=%.6f CE=%.6f Acc=%.4f mismatch=%.4f",
            seed_idx,
            payload.get("score"),
            metrics["mean_kl"],
            metrics["mean_cross_entropy"],
            metrics["top1_accuracy"],
            metrics["mismatch_rate"],
        )

    overall = _overall_metrics(seed_rows)

    report = {
        "round": {
            "id": selected_round_id,
            "round_number": round_row.get("round_number"),
            "status": round_row.get("status"),
            "round_score": round_row.get("round_score"),
            "seed_scores": round_row.get("seed_scores"),
            "queries_used": round_row.get("queries_used"),
            "queries_max": round_row.get("queries_max"),
            "rank": round_row.get("rank"),
            "total_teams": round_row.get("total_teams"),
            "seeds_submitted": round_row.get("seeds_submitted"),
        },
        "overall_metrics": overall,
        "per_seed_metrics": seed_rows,
    }

    summary_path = out_dir / "analysis_summary.json"
    raw_path = out_dir / "analysis_raw.json"

    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    raw_path.write_text(json.dumps(seed_raw), encoding="utf-8")

    logger.info("Saved summary to %s", summary_path)
    logger.info("Saved raw analysis tensors to %s", raw_path)

    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-round analyzer using /my-rounds + /analysis/{round_id}/{seed_index}."
    )
    parser.add_argument(
        "--round-id",
        type=str,
        default=None,
        help="Specific round id to analyze. Defaults to latest scoring/completed round.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.CACHE_DIR / "post_round",
        help="Output directory for saved reports.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_post_round_analysis(args.round_id, args.out_dir)
    except Exception as exc:
        logger.error("Post-round analysis failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
