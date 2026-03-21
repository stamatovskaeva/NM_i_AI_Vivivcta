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
import logging
import sys
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

def run_observe(client: AstarClient, model: WorldModel, info: RoundInfo) -> None:
    """
    Execute the two-phase observation strategy and persist all results.

    Phase 1: tile every seed's full map (9 viewports × 5 seeds = 45 queries).
    Phase 2: spend remaining budget on the highest-uncertainty regions.
    """
    budget = client.get_budget()
    cache = ObservationCache()
    planner = ObservationPlanner(
        num_seeds   = info.seeds_count,
        map_width   = info.map_width,
        map_height  = info.map_height,
        total_budget= budget.queries_remaining,
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
        planner.budget.consume()
        model.save()                # save after every query — safe to interrupt

    logger.info(model.summary())

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    if planner.budget.remaining > 0:
        logger.info(
            "Phase 2: %d queries remaining — targeting high-entropy cells",
            planner.budget.remaining,
        )
        seed_cycler: list[int] = list(range(info.seeds_count)) * planner.budget.remaining
        already_used: set[tuple[int, int, int]] = set()

        for seed_id in seed_cycler:
            if not planner.budget.can_query():
                break
            entropy_map = model.cell_entropy(seed_id)
            phase2_vps  = planner.phase2_viewports(entropy_map, seed_id)

            for vp in phase2_vps:
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
                planner.budget.consume()
                model.save()

    logger.info("Observation complete.  %s", planner.budget)
    logger.info(model.summary())


# ─────────────────────────────────────────────────────────────────────────────
# Prediction phase
# ─────────────────────────────────────────────────────────────────────────────

def run_predict(model: WorldModel) -> dict[int, np.ndarray]:
    """Build prediction tensors from the accumulated observations."""
    logger.info("Building prediction tensors …")
    predictions = model.all_predictions()
    for seed_id, tensor in predictions.items():
        _log_prediction_stats(seed_id, tensor)
    model.save_predictions()
    return predictions


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


# ─────────────────────────────────────────────────────────────────────────────
# Submission phase
# ─────────────────────────────────────────────────────────────────────────────

def run_submit(
    client: AstarClient,
    round_id: str,
    predictions: dict[int, np.ndarray],
) -> None:
    logger.info("Submitting predictions for round %s …", round_id)
    results = client.submit_all_predictions(round_id, predictions)
    logger.info("All seeds submitted: %s", results)


def run_baseline_submit(client: AstarClient, info: RoundInfo, model: WorldModel) -> None:
    """Submit a no-query baseline built from initial-state priors only."""
    logger.info("Building and submitting baseline from priors (no simulation queries).")
    predictions = run_predict(model)
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
    cmap = mcolors.ListedColormap(CLASS_COLORS)

    n = model.num_seeds
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle("Astar Island — Prediction overview", fontsize=14)

    for s in range(n):
        # Top row: dominant terrain class.
        dominant = predictions[s].argmax(axis=2)
        axes[0, s].imshow(dominant, cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
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
        choices=["all", "observe", "predict", "submit", "status", "visualise", "baseline-submit"],
        default="all",
        help=(
            "Which phase to run. 'all' runs observe → predict → submit. "
            "'baseline-submit' skips observe and submits prior-only predictions."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Load existing observations from cache instead of querying from scratch.",
    )
    p.add_argument(
        "--allow-submit",
        action="store_true",
        help="Required safety flag: explicitly allow submission to the API.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── Load or initialise the world model ──────────────────────────────────
    if args.resume and config.OBS_FILE.exists():
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
        run_observe(client, model, info)

    if args.phase in ("all", "predict", "visualise", "submit"):
        predictions = run_predict(model)

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
