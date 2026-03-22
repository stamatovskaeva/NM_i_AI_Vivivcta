"""
Configuration for Astar Island — Viking Civilisation Prediction.

All values can be overridden via a .env file (copy .env.example → .env).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── API ──────────────────────────────────────────────────────────────────────
API_BASE   = os.getenv("API_BASE",        "https://api.ainm.no/astar-island")
AUTH_TOKEN = os.getenv("ASTAR_API_TOKEN", "")

# ── Round / map parameters ───────────────────────────────────────────────────
# Defaults match the documented 40×40 map; overridden at runtime via round info.
MAP_WIDTH           = int(os.getenv("MAP_WIDTH",  "40"))
MAP_HEIGHT          = int(os.getenv("MAP_HEIGHT", "40"))
NUM_TERRAIN_CLASSES = 6      # prediction classes 0‥5
VIEWPORT_MAX_SIZE   = 15     # hard cap (API allows 5–15)
VIEWPORT_MIN_SIZE   = 5      # API minimum
TOTAL_QUERY_BUDGET  = 50     # shared across all seeds per round
NUM_SEEDS           = 5      # ground-truth seeds to predict

# ── Terrain code → prediction class mapping ──────────────────────────────────
# Raw grid values returned by the API (initial_states + simulate):
#   0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain,
#   10=Ocean, 11=Plains
# Ocean/plains/empty collapse into prediction class 0 for API submission.
TERRAIN_TO_CLASS: dict[int, int] = {
    0: 0,   # Empty
    1: 1,   # Settlement
    2: 2,   # Port
    3: 3,   # Ruin
    4: 4,   # Forest
    5: 5,   # Mountain
    10: 0,  # Ocean
    11: 0,  # Plains
}

# ── Probability model ────────────────────────────────────────────────────────
# Dirichlet prior α: added to every class count before normalising.
# A small value (< 1) gives near-uniform smoothing for cells with few
# observations while still being dominated by actual samples.
DIRICHLET_ALPHA = 0.068

# Temperature applied to observed-cell posterior probabilities.
# 1.0 = no change, <1.0 sharpens, >1.0 softens.
OBSERVED_POSTERIOR_TEMPERATURE = 1.20

# Minimum probability floor — CRITICAL for scoring.
# KL divergence → ∞ when prediction has 0 where ground truth has >0.
# Enforcing a floor of 0.01 per class prevents score destruction.
PROB_FLOOR = 0.001

# Gaussian spatial‐smoothing σ (in cells) used when interpolating
# unobserved cells.  Smaller σ → only very close neighbours contribute.
INTERP_SIGMA = 3.0

# Blend weight for prior vs interpolation on unobserved cells.
# 0.0 => interpolation only, 1.0 => prior only.
# IMPORTANT: Lower blend means trust observed data more, heuristic priors less.
# R15 likely relied heavily on actual observations rather than coast/forest heuristics.
PRIOR_BLEND = 0.40  # was 0.65 — reduce prior influence

# Confidence assigned to static cells (Ocean/Mountain) before floor/renormalize.
STATIC_CLASS_CONFIDENCE = 0.95

# Owner-cluster prior: Gaussian sigma (cells) for spreading settlement density
# across faction territory; boost magnitude added to class-1 prior on unobserved
# land cells inside the cluster.
OWNER_CLUSTER_SIGMA = 4.0    # broader than INTERP_SIGMA — faction territory is larger
OWNER_CLUSTER_BOOST = 0.12   # max class-1 probability lift per cell

# Terrain-informed prior feature strengths.
COAST_PROXIMITY_BOOST = 0.10         # class-2 (port) lift near coast
COAST_SETTLEMENT_BOOST = 0.06        # class-1 (settlement) lift near coast
FOREST_FRONTIER_SETTLEMENT_BOOST = 0.07  # class-1 lift near forest edge
FOREST_FRONTIER_FOREST_BOOST = 0.04      # class-4 lift near forest edge

# Phase-2 composite query scoring weights.
# IMPORTANT: Round 15 (65.69 score, 98% accuracy) likely used PURE entropy.
# Current multi-factor approach (entropy 0.60 + heuristics 0.40) introduces noise.
# Test single-factor entropy-only strategy to recover accuracy.
PHASE2_WEIGHT_ENTROPY = 1.00  # PURE entropy focus (was 0.60)
PHASE2_WEIGHT_COAST = 0.00   # (was 0.20)
PHASE2_WEIGHT_FOREST = 0.00  # (was 0.12)
PHASE2_WEIGHT_CONFLICT = 0.00 # (was 0.08)

# Correction-pass query scoring weights.
# IMPORTANT: Test if pure coverage focus recovers R15-level accuracy.
# Previous volatility weighting might be chasing false signals.
CORRECTION_WEIGHT_VOLATILITY = 0.00   # DISABLE volatility re-query (was 0.25)
CORRECTION_WEIGHT_COVERAGE   = 1.00   # PURE coverage focus (was 0.75)

# Adaptive correction policy:
# If seed coverage is below target, force stronger coverage expansion.
CORRECTION_COVERAGE_TARGET = 0.88
CORRECTION_WEIGHT_COVERAGE_LOW_COV = 0.85
# Per-class volatility risk (derived from Round 16 transition analysis)
CORRECTION_VOLATILITY_SETTLEMENT = 1.00  # highest churn: 53 transitions
CORRECTION_VOLATILITY_FOREST     = 0.80  # high churn:    48 transitions
CORRECTION_VOLATILITY_PLAINS     = 0.50  # moderate:      57 transitions (new settlements appear)
CORRECTION_VOLATILITY_RUIN       = 0.20  # low churn
CORRECTION_VOLATILITY_PORT       = 0.30  # low churn
CORRECTION_VOLATILITY_STATIC     = 0.00  # Ocean / Mountain never changed

# Holdout evaluator / auto-tuning defaults.
HOLDOUT_RATIO = 0.20
HOLDOUT_SEED = 42
TUNE_MAX_EVALS = 24

# ── Query budget allocation ──────────────────────────────────────────────────
# Phase 1: tile the full map once per seed (9 non-overlapping 15×15 windows).
# Phase 2: spend remaining queries revisiting high-uncertainty cells.
TILING_QUERIES_PER_SEED = 9   # ceil(40/15)² = 3×3

# Minimum spacing (in top-left anchor cells) between selected Phase-2
# viewports for the same seed. Helps avoid near-duplicate windows.
PHASE2_MIN_ANCHOR_SPACING = 6

# Two-phase observation split (30 + 20 strategy):
#   Early observe  → --phase observe --query-limit 30   (round start)
#   Correction run → --phase observe-correct            (at T-60 min)
PHASE1_QUERY_BUDGET = 30   # queries to spend at round start

# ── Persistence ──────────────────────────────────────────────────────────────
CACHE_DIR  = Path(__file__).parent / "cache"
OBS_FILE   = CACHE_DIR / "observations.json"
PRED_FILE  = CACHE_DIR / "predictions.npz"
OBS_QUERIES_FILE = CACHE_DIR / "query_observations.jsonl"
TUNING_REPORT_FILE = CACHE_DIR / "tuning_report.json"
TUNED_PARAMS_FILE = CACHE_DIR / "tuned_params.json"
PRED_META_FILE = CACHE_DIR / "prediction_metadata.json"
