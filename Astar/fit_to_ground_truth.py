"""Search for parameter settings that maximize score on a completed round.

This uses cached query observations plus /analysis ground truth and is intended
for retrospective analysis only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

import config
from api_client import AstarClient, SimSettlement
from compare_approaches import _fetch_ground_truth, _load_round_records, _temporary_config, _weighted_kl_and_score
from world_model import WorldModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit parameters to a closed round ground truth.")
    parser.add_argument("--round-id", required=True, help="Completed round id")
    parser.add_argument("--trials", type=int, default=200, help="Number of random trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cache-file",
        default=str(config.OBS_QUERIES_FILE),
        help="Path to query observation cache",
    )
    parser.add_argument(
        "--save-csv",
        default="",
        help="Optional CSV path for all trial results",
    )
    return parser.parse_args()


def _sample_params(rng: np.random.Generator) -> dict[str, float]:
    # Bounds chosen to keep distributions numerically stable.
    return {
        "OBSERVED_POSTERIOR_TEMPERATURE": float(rng.uniform(0.85, 1.25)),
        "DIRICHLET_ALPHA": float(rng.uniform(0.01, 0.25)),
        "PROB_FLOOR": float(10 ** rng.uniform(-4.0, -1.7)),  # ~1e-4 .. 0.02
    }


def _write_trials_csv(path: Path, round_id: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "round_id",
                "trial",
                "mean_score",
                "mean_kl",
                "mean_acc_pct",
                "params_json",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    round_id,
                    row["trial"],
                    f"{row['mean_score']:.6f}",
                    f"{row['mean_kl']:.6f}",
                    f"{row['mean_acc'] * 100.0:.6f}",
                    json.dumps(row["params"], sort_keys=True),
                ]
            )


def _build_base_model(client: AstarClient, round_id: str, records: list[dict]) -> WorldModel:
    info = client.get_round(round_id)
    model = WorldModel(
        map_height=info.map_height,
        map_width=info.map_width,
        num_seeds=info.seeds_count,
    )
    if info.initial_states:
        model.set_initial_states(info.initial_states)

    for rec in records:
        sid = int(rec["seed_index"])
        resp = rec.get("response", {})
        vp = resp.get("viewport", {})
        x = int(vp.get("x", rec.get("request", {}).get("x", 0)))
        y = int(vp.get("y", rec.get("request", {}).get("y", 0)))

        grid = np.array(resp["grid"], dtype=np.int32)
        settlements = [
            SimSettlement(
                x=int(s["x"]),
                y=int(s["y"]),
                population=float(s.get("population", 1.0)),
                food=float(s.get("food", 0.0)),
                wealth=float(s.get("wealth", 0.0)),
                defense=float(s.get("defense", 0.0)),
                has_port=bool(s.get("has_port", False)),
                alive=bool(s.get("alive", True)),
                owner_id=int(s.get("owner_id", -1)),
            )
            for s in resp.get("settlements", [])
        ]
        model.record(sid, x, y, grid)
        model.record_settlements(sid, settlements)

    return model


def _evaluate_on_ground_truth(
    base_model: WorldModel,
    gt_by_seed: dict[int, np.ndarray],
    overrides: dict[str, float],
) -> dict:
    with _temporary_config(overrides):
        model = copy.deepcopy(base_model)
        preds = model.all_predictions()

    per_seed = []
    for seed_idx, gt in gt_by_seed.items():
        pred = preds[seed_idx]
        kl, score = _weighted_kl_and_score(gt, pred)
        acc = float((pred.argmax(axis=2) == gt.argmax(axis=2)).mean())
        per_seed.append({"seed": seed_idx, "kl": kl, "score": score, "acc": acc})

    return {
        "mean_score": float(np.mean([p["score"] for p in per_seed])),
        "mean_kl": float(np.mean([p["kl"] for p in per_seed])),
        "mean_acc": float(np.mean([p["acc"] for p in per_seed])),
    }


def main() -> int:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    client = AstarClient()
    records = _load_round_records(Path(args.cache_file), args.round_id)
    gt_by_seed, submitted_scores = _fetch_ground_truth(client, args.round_id)
    base_model = _build_base_model(client, args.round_id, records)
    submitted_mean = float(np.nanmean(list(submitted_scores.values())))

    best = None
    rows: list[dict] = []

    # Include current config as baseline trial 0.
    baseline_params = {
        "OBSERVED_POSTERIOR_TEMPERATURE": float(getattr(config, "OBSERVED_POSTERIOR_TEMPERATURE", 1.0)),
        "DIRICHLET_ALPHA": float(getattr(config, "DIRICHLET_ALPHA", 0.1)),
        "PROB_FLOOR": float(getattr(config, "PROB_FLOOR", 0.01)),
    }
    baseline = _evaluate_on_ground_truth(base_model, gt_by_seed, baseline_params)
    baseline_row = {
        "trial": 0,
        "mean_score": baseline["mean_score"],
        "mean_kl": baseline["mean_kl"],
        "mean_acc": baseline["mean_acc"],
        "params": baseline_params,
    }
    rows.append(baseline_row)
    best = baseline_row

    for trial in range(1, args.trials + 1):
        params = _sample_params(rng)
        out = _evaluate_on_ground_truth(base_model, gt_by_seed, params)
        row = {
            "trial": trial,
            "mean_score": out["mean_score"],
            "mean_kl": out["mean_kl"],
            "mean_acc": out["mean_acc"],
            "params": params,
        }
        rows.append(row)
        if row["mean_score"] > best["mean_score"]:
            best = row

    rows.sort(key=lambda r: r["mean_score"], reverse=True)

    print("=" * 100)
    print(f"FIT TO GROUND TRUTH  round_id={args.round_id}")
    print(f"records={len(records)}  trials={args.trials}  seed={args.seed}")
    print(f"submitted_mean_score={submitted_mean:.4f}")
    print("=" * 100)
    print("Top 10:")
    for i, row in enumerate(rows[:10], start=1):
        delta_vs_sub = row["mean_score"] - submitted_mean
        print(
            f"{i:2d}. trial={row['trial']:3d}  score={row['mean_score']:.4f}  "
            f"Δ_vs_sub={delta_vs_sub:+.4f}  kl={row['mean_kl']:.5f}  "
            f"acc={row['mean_acc']*100:.2f}%  params={row['params']}"
        )

    if args.save_csv:
        _write_trials_csv(Path(args.save_csv), args.round_id, rows)
        print(f"\nSaved trial CSV to {args.save_csv}")

    print("\nBest parameters:")
    print(best["params"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
