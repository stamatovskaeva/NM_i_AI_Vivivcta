"""Compare multiple prediction parameter variants on one completed round.

The script replays raw query observations from ``cache/query_observations.jsonl``
into a fresh ``WorldModel`` and evaluates each config variant against ground
truth from ``/analysis/{round_id}/{seed_index}``.

Example:
    python compare_approaches.py --round-id <ROUND_ID>

Custom variant example:
    python compare_approaches.py --round-id <ROUND_ID> \
      --variant entropy_only:PHASE2_WEIGHT_ENTROPY=1.0,PHASE2_WEIGHT_COAST=0.0,PHASE2_WEIGHT_FOREST=0.0,PHASE2_WEIGHT_CONFLICT=0.0,PRIOR_BLEND=0.40
"""
from __future__ import annotations

import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import config
from api_client import AstarClient, SimSettlement
from world_model import WorldModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one round and compare prediction variants.")
    parser.add_argument("--round-id", required=True, help="Round ID to replay/evaluate.")
    parser.add_argument(
        "--cache-file",
        default=str(config.OBS_QUERIES_FILE),
        help="Path to query_observations.jsonl (default: config.OBS_QUERIES_FILE).",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help=(
            "Optional variant override in format "
            "name:KEY=VAL,KEY=VAL. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--save-csv",
        default="",
        help="Optional output path for CSV summary table.",
    )
    return parser.parse_args()


def _default_variants() -> dict[str, dict[str, float]]:
    return {
        "current": {},
        "entropy_only": {
            "PHASE2_WEIGHT_ENTROPY": 1.0,
            "PHASE2_WEIGHT_COAST": 0.0,
            "PHASE2_WEIGHT_FOREST": 0.0,
            "PHASE2_WEIGHT_CONFLICT": 0.0,
            "CORRECTION_WEIGHT_VOLATILITY": 0.0,
            "CORRECTION_WEIGHT_COVERAGE": 1.0,
            "PRIOR_BLEND": 0.40,
        },
        "heuristic_mix": {
            "PHASE2_WEIGHT_ENTROPY": 0.60,
            "PHASE2_WEIGHT_COAST": 0.20,
            "PHASE2_WEIGHT_FOREST": 0.12,
            "PHASE2_WEIGHT_CONFLICT": 0.08,
            "CORRECTION_WEIGHT_VOLATILITY": 0.25,
            "CORRECTION_WEIGHT_COVERAGE": 0.75,
            "PRIOR_BLEND": 0.65,
        },
    }


def _parse_variant_spec(spec: str) -> tuple[str, dict[str, float]]:
    if ":" not in spec:
        raise ValueError(f"Invalid --variant (missing ':'): {spec}")
    name, params_str = spec.split(":", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid --variant name in: {spec}")

    params: dict[str, float] = {}
    if params_str.strip():
        for pair in params_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"Invalid KEY=VAL pair '{pair}' in --variant {name}")
            key, value = pair.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Empty key in --variant {name}")
            params[key] = float(value)
    return name, params


@contextmanager
def _temporary_config(overrides: dict[str, float]):
    old_values: dict[str, float] = {}
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"config has no attribute '{key}'")
        old_values[key] = float(getattr(config, key))
        setattr(config, key, float(value))
    try:
        yield
    finally:
        for key, value in old_values.items():
            setattr(config, key, value)


def _load_round_records(cache_path: Path, round_id: str) -> list[dict]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    records: list[dict] = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("round_id") != round_id:
                continue
            records.append(row)

    if not records:
        raise RuntimeError(
            f"No cache records for round_id={round_id} in {cache_path}."
        )
    return records


def _build_model_from_cache(client: AstarClient, round_id: str, records: list[dict]) -> WorldModel:
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


def _weighted_kl_and_score(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    eps = 1e-10
    kl_per_cell = np.sum(gt * (np.log(gt + eps) - np.log(pred + eps)), axis=-1)
    entropy = -np.sum(gt * np.log(gt + eps), axis=-1)
    mask = entropy > 1e-6
    if not mask.any():
        weighted_kl = 0.0
    else:
        weighted_kl = float(np.sum(entropy[mask] * kl_per_cell[mask]) / np.sum(entropy[mask]))
    score = float(max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl))))
    return weighted_kl, score


def _fetch_ground_truth(client: AstarClient, round_id: str) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    gt_by_seed: dict[int, np.ndarray] = {}
    submitted_score: dict[int, float] = {}
    for seed_idx in range(config.NUM_SEEDS):
        analysis = client.get_analysis(round_id, seed_idx)
        gt_by_seed[seed_idx] = np.array(analysis["ground_truth"], dtype=np.float32)
        submitted_score[seed_idx] = float(analysis.get("score", np.nan))
    return gt_by_seed, submitted_score


def _evaluate_variant(
    client: AstarClient,
    round_id: str,
    records: list[dict],
    gt_by_seed: dict[int, np.ndarray],
    overrides: dict[str, float],
) -> dict:
    with _temporary_config(overrides):
        model = _build_model_from_cache(client, round_id, records)
        preds = model.all_predictions()

    per_seed = []
    for seed_idx, gt in gt_by_seed.items():
        pred = preds[seed_idx]
        kl, score = _weighted_kl_and_score(gt, pred)
        acc = float((pred.argmax(axis=2) == gt.argmax(axis=2)).mean())
        per_seed.append(
            {
                "seed": seed_idx,
                "kl": kl,
                "score": score,
                "acc": acc,
            }
        )

    mean_score = float(np.mean([p["score"] for p in per_seed]))
    mean_kl = float(np.mean([p["kl"] for p in per_seed]))
    mean_acc = float(np.mean([p["acc"] for p in per_seed]))
    return {
        "params": overrides,
        "per_seed": per_seed,
        "mean_score": mean_score,
        "mean_kl": mean_kl,
        "mean_acc": mean_acc,
    }


def _write_summary_csv(
    out_path: Path,
    round_id: str,
    submitted_mean: float,
    results: list[tuple[str, dict]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "round_id",
                "variant",
                "mean_score",
                "delta_vs_submitted",
                "mean_kl",
                "mean_acc_pct",
                "params_json",
            ]
        )
        for name, out in results:
            delta = out["mean_score"] - submitted_mean
            writer.writerow(
                [
                    round_id,
                    name,
                    f"{out['mean_score']:.6f}",
                    f"{delta:.6f}",
                    f"{out['mean_kl']:.6f}",
                    f"{out['mean_acc'] * 100.0:.6f}",
                    json.dumps(out["params"], sort_keys=True),
                ]
            )


def main() -> int:
    args = _parse_args()
    round_id = args.round_id
    cache_path = Path(args.cache_file)

    variants = _default_variants()
    for spec in args.variant:
        name, params = _parse_variant_spec(spec)
        variants[name] = params

    client = AstarClient()
    records = _load_round_records(cache_path, round_id)

    try:
        gt_by_seed, submitted_scores = _fetch_ground_truth(client, round_id)
    except Exception as exc:
        print(f"Ground truth unavailable for round {round_id}: {exc}")
        print("Run this script after round status is scoring/completed.")
        return 1

    submitted_mean = float(np.nanmean(list(submitted_scores.values())))

    print("=" * 100)
    print(f"COMPARE APPROACHES  round_id={round_id}")
    print(f"cache={cache_path}")
    print(f"records={len(records)}")
    print(f"submitted_mean_score={submitted_mean:.2f}")
    print("=" * 100)

    results: list[tuple[str, dict]] = []
    for name, overrides in variants.items():
        out = _evaluate_variant(client, round_id, records, gt_by_seed, overrides)
        results.append((name, out))

    results.sort(key=lambda x: x[1]["mean_score"], reverse=True)

    header = f"{'variant':<18} {'mean_score':>10} {'Δ_vs_sub':>10} {'mean_kl':>10} {'mean_acc':>10}"
    print(header)
    print("-" * len(header))
    for name, out in results:
        delta = out["mean_score"] - submitted_mean
        print(
            f"{name:<18} {out['mean_score']:10.2f} {delta:10.2f} {out['mean_kl']:10.4f} {out['mean_acc']*100:9.2f}%"
        )

    print("\nBest variant parameters:")
    best_name, best = results[0]
    print(f"  {best_name}: {best['params']}")

    if args.save_csv:
        out_path = Path(args.save_csv)
        _write_summary_csv(out_path, round_id, submitted_mean, results)
        print(f"\nSaved CSV summary to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
