"""
Main orchestration script for Astar Island — Viking Civilisation Prediction.

Usage
-----
    # Run everything end-to-end
    python main.py

    # Individual phases (useful for resuming after interruption)
    python main.py --phase observe
    python main.py --phase predict
    python main.py --phase submit

    # Inspect current model without running any queries
    python main.py --phase status

    # Visualise the accumulated model (requires matplotlib)
    python main.py --phase visualise
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config
from api_client import AstarClient, RoundInfo
from observation_cache import ObservationCache
from strategy import ObservationPlanner, Viewport
from world_model import WorldModel

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Observation phase
# ─────────────────────────────────────────────────────────────────────────────

def run_observe(
    client: AstarClient,
    model: WorldModel,
    info: RoundInfo,
    query_limit: int | None = None,
) -> None:
    """
    Execute the two-phase observation strategy and persist all results.

    Phase 1: tile every seed's full map (9 viewports × 5 seeds = 45 queries).
    Phase 2: spend remaining budget on the highest-uncertainty regions.

    Args:
        query_limit: If set, cap the number of queries this run may use.
                     Pass ``config.PHASE1_QUERY_BUDGET`` (30) for the early
                     batch and leave ``None`` for the full run.
    """
    budget = client.get_budget()
    effective_budget = budget.queries_remaining
    if query_limit is not None:
        effective_budget = min(effective_budget, query_limit)
    cache = ObservationCache()
    planner = ObservationPlanner(
        num_seeds   = info.seeds_count,
        map_width   = info.map_width,
        map_height  = info.map_height,
        total_budget= effective_budget,
    )

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    phase1 = planner.phase1_viewports()
    logger.info("Phase 1: %d viewports (full-map tiling, %d queries remaining)",
                len(phase1), planner.budget.remaining)

    for vp in tqdm(phase1, desc="Phase 1 — tiling", unit="query"):
        if not planner.budget.can_query():
            logger.warning("Budget exhausted during Phase 1; stopping early.")
            break
        result = client.simulate(
            round_id   = info.round_id,
            seed_index = vp.seed_id,
            vp_x=vp.x, vp_y=vp.y,
            vp_w=vp.width, vp_h=vp.height,
        )
        cache.append(
            round_id=info.round_id,
            seed_index=vp.seed_id,
            requested_x=vp.x,
            requested_y=vp.y,
            requested_w=vp.width,
            requested_h=vp.height,
            result=result,
        )
        model.record(vp.seed_id, result.viewport_x, result.viewport_y, result.grid)
        model.record_settlements(vp.seed_id, result.settlements)
        planner.budget.consume()
        model.save()                # save after every query — safe to interrupt

    logger.info(model.summary())

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    if planner.budget.remaining > 0:
        # Rank seeds by mean entropy (highest first) to prioritize least-understood ones
        seed_rank = model.seed_rank_by_entropy()
        allocation = planner.allocate_by_seed_entropy(seed_rank)
        
        logger.info("Phase 2: %d queries remaining — strategic allocation (entropy+terrain+conflict)", planner.budget.remaining)
        for seed_id in seed_rank:
            ent = model.mean_entropy(seed_id)
            logger.info(f"  seed {seed_id}: entropy={ent:.4f}, will query {allocation[seed_id]}× if budget allows")
        
        already_used: set[tuple[int, int, int]] = set()
        queries_per_seed = {sid: 0 for sid in range(info.seeds_count)}

        for seed_id in seed_rank:
            if not planner.budget.can_query():
                break
            # Only query this seed up to its allocated share
            if queries_per_seed[seed_id] >= allocation[seed_id]:
                continue
            score_map = model.strategic_query_score(seed_id)
            phase2_vps  = planner.phase2_viewports(score_map, seed_id)

            for vp in phase2_vps:
                if queries_per_seed[seed_id] >= allocation[seed_id]:
                    break
                key = (vp.seed_id, vp.x, vp.y)
                if key in already_used:
                    continue
                if not planner.budget.can_query():
                    break
                already_used.add(key)
                result = client.simulate(
                    round_id   = info.round_id,
                    seed_index = vp.seed_id,
                    vp_x=vp.x, vp_y=vp.y,
                    vp_w=vp.width, vp_h=vp.height,
                )
                cache.append(
                    round_id=info.round_id,
                    seed_index=vp.seed_id,
                    requested_x=vp.x,
                    requested_y=vp.y,
                    requested_w=vp.width,
                    requested_h=vp.height,
                    result=result,
                )
                model.record(vp.seed_id, result.viewport_x, result.viewport_y, result.grid)
                model.record_settlements(vp.seed_id, result.settlements)
                planner.budget.consume()
                queries_per_seed[seed_id] += 1
                model.save()

    logger.info("Observation complete.  %s", planner.budget)
    logger.info(model.summary())


def run_observe_correct(
    client: AstarClient,
    model: WorldModel,
    info: RoundInfo,
) -> None:
    """
    Correction observation pass (T-60 min): skip Phase 1, spend ALL remaining
    API budget as targeted Phase 2 queries on highest-uncertainty cells.

    Designed to be called after an initial ``run_observe(..., query_limit=35)``
    followed by a tune+predict+submit cycle.  Auto-resume from cache is
    expected before calling this.
    """
    budget = client.get_budget()
    if budget.queries_remaining == 0:
        logger.info("Correction pass: no queries remaining — skipping.")
        return

    cache = ObservationCache()
    planner = ObservationPlanner(
        num_seeds   = info.seeds_count,
        map_width   = info.map_width,
        map_height  = info.map_height,
        total_budget= budget.queries_remaining,
    )

    logger.info(
        "Correction pass (Phase 2 only): %d queries remaining — targeting high-entropy/terrain cells.",
        planner.budget.remaining,
    )

    # Build fresh terrain priors so the score maps are up-to-date.
    if info.initial_states:
        model.set_initial_states(info.initial_states)

    seed_rank = model.seed_rank_by_entropy()
    allocation = planner.allocate_by_seed_entropy(seed_rank)
    for seed_id in seed_rank:
        ent = model.mean_entropy(seed_id)
        logger.info("  seed %d: entropy=%.4f, correction queries=%d",
                    seed_id, ent, allocation[seed_id])

    already_used: set[tuple[int, int, int]] = set()
    queries_per_seed = {sid: 0 for sid in range(info.seeds_count)}

    for seed_id in seed_rank:
        if not planner.budget.can_query():
            break
        if queries_per_seed[seed_id] >= allocation[seed_id]:
            continue
        score_map = model.correction_query_score(seed_id)
        phase2_vps = planner.phase2_viewports(score_map, seed_id)

        for vp in phase2_vps:
            if queries_per_seed[seed_id] >= allocation[seed_id]:
                break
            key = (vp.seed_id, vp.x, vp.y)
            if key in already_used:
                continue
            if not planner.budget.can_query():
                break
            already_used.add(key)
            result = client.simulate(
                round_id   = info.round_id,
                seed_index = vp.seed_id,
                vp_x=vp.x, vp_y=vp.y,
                vp_w=vp.width, vp_h=vp.height,
            )
            cache.append(
                round_id=info.round_id,
                seed_index=vp.seed_id,
                requested_x=vp.x,
                requested_y=vp.y,
                requested_w=vp.width,
                requested_h=vp.height,
                result=result,
            )
            model.record(vp.seed_id, result.viewport_x, result.viewport_y, result.grid)
            model.record_settlements(vp.seed_id, result.settlements)
            planner.budget.consume()
            queries_per_seed[seed_id] += 1
            model.save()

    logger.info("Correction pass complete.  %s", planner.budget)
    logger.info(model.summary())


# ─────────────────────────────────────────────────────────────────────────────
# Prediction phase
# ─────────────────────────────────────────────────────────────────────────────

def _tunable_param_keys() -> list[str]:
    return [
        "PRIOR_BLEND",
        "OWNER_CLUSTER_BOOST",
        "COAST_PROXIMITY_BOOST",
        "COAST_SETTLEMENT_BOOST",
        "FOREST_FRONTIER_SETTLEMENT_BOOST",
        "FOREST_FRONTIER_FOREST_BOOST",
    ]


def _current_tuned_params() -> dict[str, float]:
    return {k: float(getattr(config, k)) for k in _tunable_param_keys()}


def _apply_tuned_params(params: dict[str, float], source: str) -> dict[str, float]:
    applied: dict[str, float] = {}
    for key in _tunable_param_keys():
        if key not in params:
            continue
        value = float(params[key])
        setattr(config, key, value)
        applied[key] = value
    if applied:
        logger.info("Applied tuned parameters from %s: %s", source, applied)
    return applied


def _load_persisted_tuned_params(round_id: str | None = None) -> tuple[dict[str, float], str | None]:
    sources = [
        (config.TUNED_PARAMS_FILE, "params"),
        (config.TUNING_REPORT_FILE, "best_params"),
    ]
    for path, field in sources:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read tuned parameter file %s: %s", path, exc)
            continue

        payload_round_id = payload.get("round_id")
        if round_id and payload_round_id and payload_round_id != round_id:
            logger.warning(
                "Ignoring tuned parameters from %s because round_id=%s does not match active round %s.",
                path,
                payload_round_id,
                round_id,
            )
            continue

        params = payload.get(field, {})
        if not isinstance(params, dict) or not params:
            continue
        return ({k: float(v) for k, v in params.items()}, str(path))

    return {}, None


def _write_tuned_params(round_id: str, params: dict[str, float], holdout_ratio: float, holdout_seed: int) -> None:
    payload = {
        "round_id": round_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": {k: float(v) for k, v in params.items()},
        "holdout_ratio": float(holdout_ratio),
        "holdout_seed": int(holdout_seed),
    }
    config.TUNED_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.TUNED_PARAMS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Persisted tuned parameters to %s", config.TUNED_PARAMS_FILE)


def _write_prediction_metadata(round_id: str | None, params: dict[str, float]) -> None:
    payload = {
        "round_id": round_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_file": str(config.PRED_FILE),
        "params": {k: float(v) for k, v in params.items()},
    }
    config.PRED_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.PRED_META_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved prediction metadata to %s", config.PRED_META_FILE)


def _load_prediction_metadata() -> dict:
    if not config.PRED_META_FILE.exists():
        return {}
    try:
        return json.loads(config.PRED_META_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read prediction metadata %s: %s", config.PRED_META_FILE, exc)
        return {}


def _ensure_submission_consistency(round_id: str) -> dict[str, float]:
    current_params = _current_tuned_params()
    persisted_params, persisted_source = _load_persisted_tuned_params(round_id)
    prediction_meta = _load_prediction_metadata()

    if persisted_params and current_params != persisted_params:
        raise RuntimeError(
            "Refusing to submit because active parameters do not match persisted tuned parameters "
            f"from {persisted_source}. Active={current_params} persisted={persisted_params}"
        )

    if persisted_params:
        meta_params = prediction_meta.get("params")
        meta_round_id = prediction_meta.get("round_id")
        if not meta_params:
            raise RuntimeError(
                "Refusing to submit because tuned parameters exist but prediction metadata is missing. "
                "Run predict again before submit."
            )
        meta_params = {k: float(v) for k, v in meta_params.items()}
        if meta_round_id and meta_round_id != round_id:
            raise RuntimeError(
                "Refusing to submit because prediction metadata belongs to a different round. "
                f"prediction_round={meta_round_id} active_round={round_id}"
            )
        if meta_params != current_params:
            raise RuntimeError(
                "Refusing to submit because predictions were generated with different parameters. "
                f"prediction_params={meta_params} active_params={current_params}"
            )

    logger.info("Submission parameter check passed: %s", current_params)
    return current_params


def run_predict(model: WorldModel, round_id: str | None = None) -> dict[int, np.ndarray]:
    """Build prediction tensors from the accumulated observations."""
    persisted_params, source = _load_persisted_tuned_params(round_id)
    if persisted_params and source is not None:
        _apply_tuned_params(persisted_params, source)

    _replay_settlement_health(model, round_id)
    logger.info("Building prediction tensors …")
    predictions = model.all_predictions()
    for seed_id, tensor in predictions.items():
        _log_prediction_stats(seed_id, tensor)
    model.save_predictions()
    _write_prediction_metadata(round_id, _current_tuned_params())
    return predictions


def _replay_settlement_health(model: WorldModel, round_id: str | None = None) -> None:
    """
    Replay settlement health attributes from the observation cache into the
    model so that ruin-risk scores are populated even when ``--phase predict``
    is run standalone (without re-running observe).
    """
    cache_file = config.OBS_QUERIES_FILE
    if not cache_file.exists():
        return
    model.reset_dynamic_signals()
    n = 0
    for line in cache_file.open("r", encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if round_id and d.get("round_id") != round_id:
            continue
        sid  = d.get("seed_index")
        resp = d.get("response", {})
        if sid is None:
            continue
        settlements = [
            type("S", (), {
                "x":          s["x"],
                "y":          s["y"],
                "food":       float(s.get("food", 0)),
                "defense":    float(s.get("defense", 0)),
                "population": float(s.get("population", 1.0)),
                "alive":      bool(s.get("alive", True)),
                "has_port":   bool(s.get("has_port", False)),
                "owner_id":   int(s.get("owner_id", -1)),
            })()
            for s in resp.get("settlements", [])
        ]
        model.record_settlements(sid, settlements)
        n += len(settlements)
    if n:
        logger.info("Replayed %d settlement health observations into ruin-risk map.", n)
        for sid in range(model.num_seeds):
            rr = model._ruin_risk[sid]
            high = int((rr > 0.6).sum())
            logger.info("  seed %d: %d cells with ruin-risk, %d high-risk (>0.6)",
                        sid, int((rr > 0).sum()), high)


def _log_prediction_stats(seed_id: int, tensor: np.ndarray) -> None:
    dominant = tensor.argmax(axis=2)            # (H, W) most-likely class
    class_counts = np.bincount(dominant.ravel(), minlength=config.NUM_TERRAIN_CLASSES)
    mean_ent = float(
        -(tensor * np.where(tensor > 0, np.log(tensor), 0.0)).sum(axis=2).mean()
    )
    logger.info(
        "Seed %d  mean_entropy=%.4f  class_distribution=%s",
        seed_id,
        mean_ent,
        dict(enumerate(class_counts.tolist())),
    )


@dataclass
class HoldoutEval:
    mean_cross_entropy: float
    mean_kl: float
    top1_accuracy: float
    heldout_cells: int
    heldout_weight: float


def _build_holdout_masks(
    model: WorldModel,
    ratio: float,
    seed: int,
) -> dict[int, np.ndarray]:
    """Sample held-out observed cells per seed for offline evaluation."""
    rng = np.random.default_rng(seed)
    masks: dict[int, np.ndarray] = {}
    for sid in range(model.num_seeds):
        observed_rc = np.argwhere(model._observed[sid])
        mask = np.zeros((model.H, model.W), dtype=bool)
        if len(observed_rc) == 0:
            masks[sid] = mask
            continue
        n_holdout = int(round(len(observed_rc) * ratio))
        n_holdout = max(1, min(n_holdout, len(observed_rc)))
        pick = rng.choice(len(observed_rc), size=n_holdout, replace=False)
        rows = observed_rc[pick, 0]
        cols = observed_rc[pick, 1]
        mask[rows, cols] = True
        masks[sid] = mask
    return masks


def _evaluate_holdout(
    source_model: WorldModel,
    holdout_masks: dict[int, np.ndarray],
) -> HoldoutEval:
    """
    Evaluate predictive quality on held-out observed cells.

    - Removes held-out cells from the training view.
    - Predicts them using the same model pipeline.
    - Scores weighted cross-entropy / KL against empirical observed class mix.
    """
    eval_model = copy.deepcopy(source_model)
    for sid, mask in holdout_masks.items():
        if mask.any():
            eval_model._counts[sid][mask, :] = 0.0
            eval_model._observed[sid][mask] = False

    preds = eval_model.all_predictions()

    total_weight = 0.0
    weighted_ce = 0.0
    weighted_kl = 0.0
    weighted_acc = 0.0
    cell_count = 0

    for sid, mask in holdout_masks.items():
        if not mask.any():
            continue

        true_counts = source_model._counts[sid][mask]  # (N, K)
        totals = true_counts.sum(axis=1)
        valid = totals > 0
        if not valid.any():
            continue

        true_counts = true_counts[valid]
        totals = totals[valid]
        p_true = true_counts / totals[:, None]

        q_pred = preds[sid][mask][valid]
        q_pred = np.clip(q_pred, config.PROB_FLOOR, 1.0)
        q_pred = q_pred / q_pred.sum(axis=1, keepdims=True)

        ce = -np.sum(p_true * np.log(q_pred), axis=1)
        kl = np.sum(
            p_true * (np.log(np.clip(p_true, 1e-12, 1.0)) - np.log(q_pred)), axis=1
        )
        acc = (np.argmax(p_true, axis=1) == np.argmax(q_pred, axis=1)).astype(np.float32)

        weighted_ce += float(np.sum(ce * totals))
        weighted_kl += float(np.sum(kl * totals))
        weighted_acc += float(np.sum(acc * totals))
        total_weight += float(np.sum(totals))
        cell_count += int(len(totals))

    if total_weight <= 0:
        return HoldoutEval(
            mean_cross_entropy=float("nan"),
            mean_kl=float("nan"),
            top1_accuracy=float("nan"),
            heldout_cells=0,
            heldout_weight=0.0,
        )

    return HoldoutEval(
        mean_cross_entropy=weighted_ce / total_weight,
        mean_kl=weighted_kl / total_weight,
        top1_accuracy=weighted_acc / total_weight,
        heldout_cells=cell_count,
        heldout_weight=total_weight,
    )


def run_tune(
    model: WorldModel,
    info: RoundInfo,
    holdout_ratio: float,
    holdout_seed: int,
    max_evals: int,
) -> None:
    """Run holdout evaluation + lightweight auto-tuning over key model weights."""
    holdout_ratio = float(np.clip(holdout_ratio, 0.05, 0.60))
    max_evals = max(4, int(max_evals))

    if not any(model._observed[s].any() for s in range(model.num_seeds)):
        logger.error("Tune requires cached observations. Run observe first or pass --resume with a valid cache.")
        return

    _replay_settlement_health(model, info.round_id)
    holdout_masks = _build_holdout_masks(model, holdout_ratio, holdout_seed)

    tune_levels: dict[str, list[float]] = {
        "PRIOR_BLEND": [0.55, 0.65, 0.75],
        "OWNER_CLUSTER_BOOST": [0.08, 0.12, 0.16],
        "COAST_PROXIMITY_BOOST": [0.06, 0.10, 0.14],
        "COAST_SETTLEMENT_BOOST": [0.04, 0.06, 0.09],
        "FOREST_FRONTIER_SETTLEMENT_BOOST": [0.05, 0.07, 0.10],
        "FOREST_FRONTIER_FOREST_BOOST": [0.02, 0.04, 0.06],
    }

    keys = list(tune_levels.keys())
    base_params = {k: float(getattr(config, k)) for k in keys}

    all_candidates = [
        dict(zip(keys, combo))
        for combo in itertools.product(*(tune_levels[k] for k in keys))
    ]

    rng = np.random.default_rng(holdout_seed)
    rng.shuffle(all_candidates)
    all_candidates = [base_params] + [c for c in all_candidates if c != base_params]
    candidates = all_candidates[:max_evals]

    best_params = base_params.copy()
    best_eval = HoldoutEval(mean_cross_entropy=np.inf, mean_kl=np.inf,
                            top1_accuracy=0.0, heldout_cells=0, heldout_weight=0.0)
    rows: list[dict] = []

    logger.info("Tuning: evaluating %d candidates (holdout_ratio=%.2f, seed=%d)",
                len(candidates), holdout_ratio, holdout_seed)

    for i, params in enumerate(candidates, start=1):
        for k, v in params.items():
            setattr(config, k, float(v))

        candidate_model = copy.deepcopy(model)
        if info.initial_states:
            candidate_model.set_initial_states(info.initial_states)

        metrics = _evaluate_holdout(candidate_model, holdout_masks)
        row = {
            "rank": i,
            "params": {k: float(v) for k, v in params.items()},
            "mean_cross_entropy": float(metrics.mean_cross_entropy),
            "mean_kl": float(metrics.mean_kl),
            "top1_accuracy": float(metrics.top1_accuracy),
            "heldout_cells": int(metrics.heldout_cells),
            "heldout_weight": float(metrics.heldout_weight),
        }
        rows.append(row)

        logger.info(
            "Tune %02d/%02d: KL=%.6f CE=%.6f Acc=%.4f params=%s",
            i,
            len(candidates),
            metrics.mean_kl,
            metrics.mean_cross_entropy,
            metrics.top1_accuracy,
            params,
        )

        if metrics.mean_kl < best_eval.mean_kl:
            best_eval = metrics
            best_params = {k: float(v) for k, v in params.items()}

    for k, v in best_params.items():
        setattr(config, k, float(v))

    if info.initial_states:
        model.set_initial_states(info.initial_states)

    logger.info("Best params by holdout KL: %s", best_params)
    logger.info(
        "Best holdout metrics: KL=%.6f CE=%.6f Acc=%.4f (cells=%d)",
        best_eval.mean_kl,
        best_eval.mean_cross_entropy,
        best_eval.top1_accuracy,
        best_eval.heldout_cells,
    )

    report = {
        "round_id": info.round_id,
        "holdout_ratio": holdout_ratio,
        "holdout_seed": holdout_seed,
        "max_evals": len(candidates),
        "best_params": best_params,
        "best_metrics": {
            "mean_kl": float(best_eval.mean_kl),
            "mean_cross_entropy": float(best_eval.mean_cross_entropy),
            "top1_accuracy": float(best_eval.top1_accuracy),
            "heldout_cells": int(best_eval.heldout_cells),
            "heldout_weight": float(best_eval.heldout_weight),
        },
        "trials": rows,
    }
    report_path = config.TUNING_REPORT_FILE
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Tuning report saved to %s", report_path)

    _write_tuned_params(info.round_id, best_params, holdout_ratio, holdout_seed)

    predictions = run_predict(model, info.round_id)
    logger.info("Predictions regenerated with tuned weights for optional submit.")


# ─────────────────────────────────────────────────────────────────────────────
# Submission phase
# ─────────────────────────────────────────────────────────────────────────────

def run_submit(
    client: AstarClient,
    round_id: str,
    predictions: dict[int, np.ndarray],
) -> None:
    active_params = _ensure_submission_consistency(round_id)
    logger.info("Submitting with parameters: %s", active_params)
    logger.info("Submitting predictions for round %s …", round_id)
    results = client.submit_all_predictions(round_id, predictions)
    logger.info("All seeds submitted: %s", results)


def run_baseline_submit(client: AstarClient, info: RoundInfo, model: WorldModel) -> None:
    """Submit a no-query baseline built from initial-state priors only."""
    logger.info("Building and submitting baseline from priors (no simulation queries).")
    predictions = run_predict(model, info.round_id)
    run_submit(client, info.round_id, predictions)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation (optional)
# ─────────────────────────────────────────────────────────────────────────────

def run_visualise(model: WorldModel, predictions: dict[int, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        logger.error("matplotlib not installed; run: pip install matplotlib")
        return

    # Colour map for 6 terrain classes.
    CLASS_COLORS = ["#1a78c2", "#c2a94a", "#2d7c32", "#a8d96e", "#c44b2b", "#888"]
    class_count = config.NUM_TERRAIN_CLASSES
    cmap = mcolors.ListedColormap(CLASS_COLORS)

    n = model.num_seeds
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle("Astar Island — Prediction overview", fontsize=14)

    for s in range(n):
        # Top row: dominant terrain class.
        dominant = predictions[s].argmax(axis=2)
        axes[0, s].imshow(
            dominant,
            cmap=cmap,
            vmin=0,
            vmax=class_count - 1,
            interpolation="nearest",
        )
        axes[0, s].set_title(f"Seed {s}\n(dominant class)")
        axes[0, s].axis("off")

        # Bottom row: prediction entropy (brighter = more uncertain).
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(predictions[s] * np.where(
                predictions[s] > 0, np.log(predictions[s]), 0.0
            )).sum(axis=2)
        im = axes[1, s].imshow(ent, cmap="hot", interpolation="nearest")
        axes[1, s].set_title("Entropy")
        axes[1, s].axis("off")
        plt.colorbar(im, ax=axes[1, s], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_dir = config.CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "visualisation.png"
    plt.savefig(save_path, dpi=150)
    logger.info("Visualisation saved to %s", save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Astar Island — Viking Civilisation Prediction pipeline"
    )
    p.add_argument(
        "--phase",
        choices=["all", "observe", "observe-correct", "predict", "submit", "status", "visualise", "baseline-submit", "tune"],
        default="all",
        help=(
            "Which phase to run. 'all' runs observe → predict → submit. "
            "'observe' runs the initial observation pass (use --query-limit 30 for the 30+20 strategy). "
            "'observe-correct' resumes from cache and spends all remaining queries as targeted corrections (run at T-60 min). "
            "'baseline-submit' skips observe and submits prior-only predictions. "
            "'tune' runs holdout evaluation + auto-weight search."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Load existing observations from cache instead of querying from scratch.",
    )
    p.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help=(
            "Cap the number of queries used by --phase observe. "
            f"Defaults to all remaining budget. "
            f"Set to {config.PHASE1_QUERY_BUDGET} for the recommended 30+20 split."
        ),
    )
    p.add_argument(
        "--allow-submit",
        action="store_true",
        help="Required safety flag: explicitly allow submission to the API.",
    )
    p.add_argument(
        "--holdout-ratio",
        type=float,
        default=config.HOLDOUT_RATIO,
        help="Fraction of observed cells to hold out during tune phase (default from config).",
    )
    p.add_argument(
        "--holdout-seed",
        type=int,
        default=config.HOLDOUT_SEED,
        help="Random seed for holdout split and candidate sampling.",
    )
    p.add_argument(
        "--tune-max-evals",
        type=int,
        default=config.TUNE_MAX_EVALS,
        help="Maximum number of weight candidates to evaluate in tune phase.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── Load or initialise the world model ──────────────────────────────────
    auto_resume_phases = {"predict", "submit", "visualise", "tune", "observe-correct"}
    should_resume = (args.resume or args.phase in auto_resume_phases) and config.OBS_FILE.exists()
    if should_resume:
        logger.info("Resuming from cached observations: %s", config.OBS_FILE)
        model = WorldModel.load(config.OBS_FILE)
    else:
        model = WorldModel()

    # ── Status-only: no API needed ───────────────────────────────────────────
    if args.phase == "status":
        print(model.summary())
        return 0

    # ── For all other phases we need an API client ───────────────────────────
    if not config.AUTH_TOKEN:
        logger.error(
            "ASTAR_API_TOKEN is not set.  Copy .env.example → .env and "
            "fill in your token from app.ainm.no."
        )
        return 1

    client = AstarClient()
    info   = client.get_active_round()

    # Resize model if the API reports different dimensions than the defaults.
    if info.map_width != model.W or info.map_height != model.H or info.seeds_count != model.num_seeds:
        logger.info(
            "Reinitialising model for map=%d×%d seeds=%d.",
            info.map_width, info.map_height, info.seeds_count,
        )
        model = WorldModel(
            map_height=info.map_height,
            map_width=info.map_width,
            num_seeds=info.seeds_count,
        )

    # Seed the model with the known initial terrain (free, no query cost).
    if info.initial_states:
        model.set_initial_states(info.initial_states)

    predictions: dict[int, np.ndarray] = {}

    if args.phase == "baseline-submit":
        if not args.allow_submit:
            logger.error(
                "Submission blocked: pass --allow-submit to explicitly consent."
            )
            return 1
        run_baseline_submit(client, info, model)
        return 0

    if args.phase in ("all", "observe"):
        run_observe(client, model, info, query_limit=args.query_limit)

    if args.phase == "observe-correct":
        run_observe_correct(client, model, info)

    if args.phase == "tune":
        run_tune(
            model=model,
            info=info,
            holdout_ratio=args.holdout_ratio,
            holdout_seed=args.holdout_seed,
            max_evals=args.tune_max_evals,
        )
        return 0

    if args.phase in ("all", "predict", "visualise", "submit"):
        predictions = run_predict(model, info.round_id)

    if args.phase == "visualise":
        run_visualise(model, predictions)
        return 0

    if args.phase in ("all", "submit"):
        if not args.allow_submit:
            logger.error(
                "Submission blocked: pass --allow-submit to explicitly consent."
            )
            return 1
        if not predictions:
            # Load from disk if we skipped the predict step this run.
            if config.PRED_FILE.exists():
                npz = np.load(config.PRED_FILE)
                predictions = {int(k): npz[k] for k in npz.files}
            else:
                logger.error(
                    "No predictions available.  Run with --phase predict first."
                )
                return 1
        run_submit(client, info.round_id, predictions)

    return 0


if __name__ == "__main__":
    sys.exit(main())
