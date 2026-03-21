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
DIRICHLET_ALPHA = 0.1

# Minimum probability floor — CRITICAL for scoring.
# KL divergence → ∞ when prediction has 0 where ground truth has >0.
# Enforcing a floor of 0.01 per class prevents score destruction.
PROB_FLOOR = 0.01

# Gaussian spatial‐smoothing σ (in cells) used when interpolating
# unobserved cells.  Smaller σ → only very close neighbours contribute.
INTERP_SIGMA = 3.0

# Blend weight for prior vs interpolation on unobserved cells.
# 0.0 => interpolation only, 1.0 => prior only.
PRIOR_BLEND = 0.65

# Confidence assigned to static cells (Ocean/Mountain) before floor/renormalize.
STATIC_CLASS_CONFIDENCE = 0.95

# Owner-cluster prior: Gaussian sigma (cells) for spreading settlement density
# across faction territory; boost magnitude added to class-1 prior on unobserved
# land cells inside the cluster.
OWNER_CLUSTER_SIGMA = 4.0    # broader than INTERP_SIGMA — faction territory is larger
OWNER_CLUSTER_BOOST = 0.12   # max class-1 probability lift per cell

# ── Query budget allocation ──────────────────────────────────────────────────
# Phase 1: tile the full map once per seed (9 non-overlapping 15×15 windows).
# Phase 2: spend remaining queries revisiting high-uncertainty cells.
TILING_QUERIES_PER_SEED = 9   # ceil(40/15)² = 3×3

# ── Persistence ──────────────────────────────────────────────────────────────
CACHE_DIR  = Path(__file__).parent / "cache"
OBS_FILE   = CACHE_DIR / "observations.json"
PRED_FILE  = CACHE_DIR / "predictions.npz"
OBS_QUERIES_FILE = CACHE_DIR / "query_observations.jsonl"
