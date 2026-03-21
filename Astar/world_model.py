"""
Probabilistic world model for Astar Island.

Observations are accumulated per (seed_id, row, col) cell.  When it is
time to predict we build a W×H×6 probability tensor for each seed using:

  1. **Observed cells** — empirical class distribution, smoothed with a
     Dirichlet prior (α = DIRICHLET_ALPHA) to avoid zero probabilities.

  2. **Unobserved cells** — Gaussian-kernel interpolation over the
     predicted distributions of nearby *observed* cells, falling back to
     the global empirical distribution when no neighbours exist.

The scoring metric is entropy-weighted KL divergence: penalising confident
wrong predictions more than uncertain ones.  Therefore we are deliberately
conservative — unobserved cells get a softer distribution rather than a
hard guess.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

import config

logger = logging.getLogger(__name__)

K = config.NUM_TERRAIN_CLASSES          # 6 prediction classes
α = config.DIRICHLET_ALPHA              # Dirichlet smoothing
_T2C = config.TERRAIN_TO_CLASS          # raw terrain code → class index


class WorldModel:
    """
    Maintains per-cell observation counts and produces prediction tensors.

    Internal storage
    ----------------
    ``_counts[seed_id]`` : ndarray of shape ``(H, W, K)``
        Raw observation counts.  Initialised to zero; incremented each time
        a viewport reveals the terrain class at a given cell.

    ``_observed[seed_id]`` : boolean ndarray of shape ``(H, W)``
        True wherever at least one observation has been recorded.
    """

    def __init__(
        self,
        map_height: int = config.MAP_HEIGHT,
        map_width:  int = config.MAP_WIDTH,
        num_seeds:  int = config.NUM_SEEDS,
    ):
        self.H = map_height
        self.W = map_width
        self.num_seeds = num_seeds

        self._counts:   dict[int, np.ndarray] = {
            s: np.zeros((self.H, self.W, K), dtype=np.float32)
            for s in range(num_seeds)
        }
        self._observed: dict[int, np.ndarray] = {
            s: np.zeros((self.H, self.W), dtype=bool)
            for s in range(num_seeds)
        }
        self._prior: dict[int, np.ndarray] = {
            s: np.full((self.H, self.W, K), 1.0 / K, dtype=np.float32)
            for s in range(num_seeds)
        }
        # Seed-specific static locks (Ocean/Mountain) from initial states.
        self._locked_class: dict[int, np.ndarray] = {
            s: np.full((self.H, self.W), -1, dtype=np.int32)
            for s in range(num_seeds)
        }

    # ── Initial state injection ─────────────────────────────────────────────────

    def set_initial_grid(self, grid: np.ndarray) -> None:
        """
        Lock permanent cells using the initial terrain grid from the round.

        Ocean (10) and Mountain (5) cells never change across 50-year
        simulations, so we can set them with near-certainty before running
        a single query.  This also improves interpolation quality for
        unobserved cells near locked anchors.

        Args:
            grid: (H, W) int array of raw terrain codes from
                  ``RoundInfo.initial_states[seed].grid``.
                  All seeds share the same base terrain, so any seed's
                  initial grid works.
        """
        locked = np.full((self.H, self.W), -1, dtype=np.int32)
        for raw_code, cls in _T2C.items():
            if raw_code in (10, 5):   # Ocean and Mountain only — truly static
                mask = grid == raw_code
                locked[mask] = cls
        for seed_id in range(self.num_seeds):
            self._locked_class[seed_id] = locked.copy()
        logger.info(
            "Locked %d static cells from initial grid "
            "(ocean+mountain never change).",
            int((locked >= 0).sum()),
        )

    def set_initial_states(self, initial_states: list) -> None:
        """Build per-seed terrain-aware priors from round initial states."""
        if not initial_states:
            return

        for seed_id in range(min(self.num_seeds, len(initial_states))):
            state = initial_states[seed_id]
            grid = np.array(state.grid, dtype=np.int32)
            prior = self._build_seed_prior(grid)

            locked = np.full((self.H, self.W), -1, dtype=np.int32)
            for raw_code, cls in _T2C.items():
                if raw_code in (10, 5):
                    mask = grid == raw_code
                    locked[mask] = cls
            self._locked_class[seed_id] = locked

            for settlement in state.settlements:
                x = int(settlement.x)
                y = int(settlement.y)
                if x < 0 or x >= self.W or y < 0 or y >= self.H:
                    continue
                prior[y, x, :] = config.PROB_FLOOR
                if settlement.has_port:
                    prior[y, x, 2] = 1.0 - (K - 1) * config.PROB_FLOOR
                else:
                    prior[y, x, 1] = 1.0 - (K - 1) * config.PROB_FLOOR

            self._prior[seed_id] = self._normalise(prior)

        logger.info("Built terrain-aware priors for %d seeds.", min(self.num_seeds, len(initial_states)))
    # ── Observation ingestion ────────────────────────────────────────────────

    def record(
        self,
        seed_id: int,
        x: int,
        y: int,
        terrain_grid: np.ndarray,
    ) -> None:
        """
        Ingest one viewport observation.

        Args:
            seed_id:      Which seed this observation belongs to.
            x, y:         Top-left corner of the viewport (col, row).
            terrain_grid: Integer array of shape ``(viewport_h, viewport_w)``
                          with **raw terrain codes** as returned by the API
                          (0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest,
                          5=Mountain, 10=Ocean, 11=Plains).
        """
        vh, vw = terrain_grid.shape
        for dr in range(vh):
            for dc in range(vw):
                row = y + dr
                col = x + dc
                if row >= self.H or col >= self.W:
                    continue
                raw = int(terrain_grid[dr, dc])
                cls = _T2C.get(raw, -1)
                if cls < 0:
                    logger.warning(
                        "Unknown terrain code %d at (%d,%d); skipping.", raw, row, col
                    )
                    continue
                self._counts[seed_id][row, col, cls] += 1.0
                self._observed[seed_id][row, col] = True

    # ── Derived statistics ───────────────────────────────────────────────────

    def coverage(self, seed_id: int) -> float:
        """Fraction of cells with at least one observation for this seed."""
        return float(self._observed[seed_id].mean())

    def cell_entropy(self, seed_id: int) -> np.ndarray:
        """
        Per-cell entropy of the current posterior, shape ``(H, W)``.
        High entropy → model is uncertain about that cell.
        """
        dist = self._posterior(seed_id)             # (H, W, K)
        # H = -Σ p·log(p),  guarded against log(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -np.sum(dist * np.where(dist > 0, np.log(dist), 0.0), axis=2)
        return ent                                   # (H, W)

    def mean_entropy(self, seed_id: int) -> float:
        return float(self.cell_entropy(seed_id).mean())

    # ── Prediction tensor ────────────────────────────────────────────────────

    def prediction_tensor(self, seed_id: int) -> np.ndarray:
        """
        Build the ``(H, W, K)`` probability tensor for one seed.

        - Observed cells: smoothed empirical distribution.
        - Unobserved cells: spatially interpolated from nearby observed cells,
          or global prior if no neighbours are reachable.
        """
        posterior = self._posterior(seed_id)        # (H, W, K)
        observed  = self._observed[seed_id]         # (H, W)

        if observed.all():
            return posterior

        # ── Gaussian kernel interpolation ───────────────────────────────────
        # For each class k, smooth the observed-cell values over the map; the
        # unobserved cells pick up a weighted combination of their neighbours'
        # distributions.  We also smooth a "weight" plane (= 1 where observed,
        # 0 elsewhere) so we can normalise correctly.

        sigma = config.INTERP_SIGMA
        interp = np.zeros((self.H, self.W, K), dtype=np.float32)
        weight_plane = observed.astype(np.float32)
        smoothed_weight = gaussian_filter(weight_plane, sigma=sigma)

        for k in range(K):
            class_plane = np.where(observed, posterior[:, :, k], 0.0).astype(np.float32)
            smoothed_class = gaussian_filter(class_plane, sigma=sigma)
            with np.errstate(invalid="ignore", divide="ignore"):
                interp[:, :, k] = np.where(
                    smoothed_weight > 0,
                    smoothed_class / smoothed_weight,
                    0.0,
                )

        # For cells where the kernel has no signal (very remote from any
        # observation), fall back to the global empirical distribution.
        global_prior = self._global_prior(seed_id)  # (K,)
        no_signal = smoothed_weight < 1e-6
        interp[no_signal] = global_prior

        # Blend: keep observed cells' own posterior; fill the rest from a
        # weighted mix of interpolation and terrain-aware priors.
        prior = self._prior[seed_id]
        blend = float(config.PRIOR_BLEND)
        unobserved_estimate = (blend * prior + (1.0 - blend) * interp).astype(np.float32)
        result = np.where(observed[:, :, np.newaxis], posterior, unobserved_estimate)

        # Apply locked cells (Ocean / Mountain) — override with near-certain dist.
        locked_map = self._locked_class[seed_id]
        locked_mask = locked_map >= 0        # (H, W)
        if locked_mask.any():
            locked_dist = np.full((self.H, self.W, K), config.PROB_FLOOR, dtype=np.float32)
            # Assign remaining probability mass to the locked class.
            for cls in range(K):
                cell_mask = locked_map == cls
                if cell_mask.any():
                    locked_dist[cell_mask, cls] = 1.0 - (K - 1) * config.PROB_FLOOR
            locked_dist = self._normalise(locked_dist)
            result = np.where(locked_mask[:, :, np.newaxis], locked_dist, result)

        # Enforce PROB_FLOOR — prevents infinite KL divergence when scoring.
        result = np.maximum(result, config.PROB_FLOOR)

        # Final renormalise to guard against floating-point drift.
        result = self._normalise(result)
        return result

    def all_predictions(self) -> dict[int, np.ndarray]:
        """Return prediction tensors for all seeds."""
        return {s: self.prediction_tensor(s) for s in range(self.num_seeds)}

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: Path = config.OBS_FILE) -> None:
        """Serialise observations to a JSON cache file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            str(s): self._counts[s].tolist() for s in range(self.num_seeds)
        }
        path.write_text(json.dumps(data))
        logger.info("Observations saved to %s", path)

    @classmethod
    def load(cls, path: Path = config.OBS_FILE, **kwargs) -> "WorldModel":
        """Restore a WorldModel from a previously saved cache."""
        model = cls(**kwargs)
        raw = json.loads(path.read_text())
        for s_str, counts_list in raw.items():
            s = int(s_str)
            counts = np.array(counts_list, dtype=np.float32)
            model._counts[s] = counts
            model._observed[s] = counts.sum(axis=2) > 0
        logger.info("Observations loaded from %s", path)
        return model

    def save_predictions(
        self, path: Path = config.PRED_FILE
    ) -> None:
        """Save prediction tensors as a compressed numpy archive."""
        path.parent.mkdir(parents=True, exist_ok=True)
        preds = self.all_predictions()
        np.savez_compressed(path, **{str(s): t for s, t in preds.items()})
        logger.info("Predictions saved to %s", path)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _posterior(self, seed_id: int) -> np.ndarray:
        """
        Dirichlet-smoothed empirical distribution, shape ``(H, W, K)``.
        Each cell sums to 1.
        """
        counts   = self._counts[seed_id] + α          # (H, W, K)
        total    = counts.sum(axis=2, keepdims=True)   # (H, W, 1)
        return (counts / total).astype(np.float32)

    def _global_prior(self, seed_id: int) -> np.ndarray:
        """
        Global empirical distribution over all observed cells for a seed.
        Falls back to uniform if nothing has been observed yet.
        """
        observed_mask = self._observed[seed_id]
        if observed_mask.any():
            total_counts = self._counts[seed_id][observed_mask].sum(axis=0)  # (K,)
            total_counts += α
            return (total_counts / total_counts.sum()).astype(np.float32)
        # If no observations, fall back to mean of terrain-aware prior.
        return self._prior[seed_id].mean(axis=(0, 1)).astype(np.float32)

    def _build_seed_prior(self, grid: np.ndarray) -> np.ndarray:
        """Construct terrain-aware prior probabilities from initial terrain."""
        prior = np.full((self.H, self.W, K), config.PROB_FLOOR, dtype=np.float32)

        def set_major(mask: np.ndarray, class_idx: int, confidence: float) -> None:
            if not mask.any():
                return
            prior[mask, :] = config.PROB_FLOOR
            prior[mask, class_idx] = confidence

        ocean = grid == 10
        mountain = grid == 5
        forest = grid == 4
        plains_or_empty = (grid == 0) | (grid == 11)

        set_major(ocean, 0, config.STATIC_CLASS_CONFIDENCE)
        set_major(mountain, 5, config.STATIC_CLASS_CONFIDENCE)

        # Forest mostly stays forest / empty mix.
        if forest.any():
            prior[forest, :] = config.PROB_FLOOR
            prior[forest, 4] = 0.74
            prior[forest, 0] = 0.20

        # Plains/empty bias toward class 0 with some dynamic tail.
        if plains_or_empty.any():
            prior[plains_or_empty, :] = config.PROB_FLOOR
            prior[plains_or_empty, 0] = 0.74
            prior[plains_or_empty, 1] = 0.10
            prior[plains_or_empty, 2] = 0.03
            prior[plains_or_empty, 3] = 0.06
            prior[plains_or_empty, 4] = 0.05

        # Coastline land has elevated chance of port/settlement.
        ocean_pad = np.pad(ocean.astype(np.int8), 1)
        coast = np.zeros_like(ocean, dtype=bool)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            coast |= ocean_pad[1 + dy : 1 + dy + self.H, 1 + dx : 1 + dx + self.W] > 0
        coast_land = coast & (~ocean) & (~mountain)
        if coast_land.any():
            prior[coast_land, :] = np.maximum(prior[coast_land, :], config.PROB_FLOOR)
            prior[coast_land, 2] = np.maximum(prior[coast_land, 2], 0.14)
            prior[coast_land, 1] = np.maximum(prior[coast_land, 1], 0.12)
            prior[coast_land, 0] = np.maximum(prior[coast_land, 0], 0.56)

        return self._normalise(prior)

    @staticmethod
    def _normalise(tensor: np.ndarray) -> np.ndarray:
        """Ensure every cell's distribution sums to 1."""
        s = tensor.sum(axis=2, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        return (tensor / s).astype(np.float32)

    # ── Reporting ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = ["WorldModel summary:"]
        for s in range(self.num_seeds):
            cov = self.coverage(s) * 100
            ent = self.mean_entropy(s)
            lines.append(f"  seed {s}: coverage={cov:.1f}%  mean_entropy={ent:.4f}")
        return "\n".join(lines)
