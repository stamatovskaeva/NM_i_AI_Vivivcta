"""
Observation strategy for Astar Island.

The 50-query budget is tight.  This module plans which viewports to query
and in what order so that we:

  Phase 1 — Full coverage:
      Tile the entire map with non-overlapping 15×15 windows.
      For a 40×40 map this takes exactly 9 queries per seed (3×3 grid),
      covering 45 queries total for 5 seeds — well within the 50-query cap.

  Phase 2 — Uncertainty refinement:
      Use the 5 remaining queries to revisit the cells whose empirical
      distributions have the highest entropy (i.e. the model is most
      uncertain about them).  The viewport is centred on the highest-
      uncertainty region found so far.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Viewport:
    """A single viewport query specification."""

    seed_id: int
    x: int
    y: int
    width: int
    height: int

    def __repr__(self) -> str:
        return (
            f"Viewport(seed={self.seed_id}, "
            f"x={self.x}, y={self.y}, "
            f"w={self.width}, h={self.height})"
        )


def tile_map(
    map_width: int = config.MAP_WIDTH,
    map_height: int = config.MAP_HEIGHT,
    viewport_size: int = config.VIEWPORT_MAX_SIZE,
) -> list[tuple[int, int, int, int]]:
    """
    Return a minimal list of (x, y, width, height) tuples that together
    cover every cell of the map exactly once (with no overlap).

    For a 40×40 map with viewport_size=15 this yields 9 windows:
        x ∈ {0, 15, 25},  y ∈ {0, 15, 25}
    The last window in each axis is clipped to the map boundary so we
    never request out-of-bounds cells.
    """
    tiles = []
    x = 0
    while x < map_width:
        y = 0
        while y < map_height:
            w = min(viewport_size, map_width  - x)
            h = min(viewport_size, map_height - y)
            tiles.append((x, y, w, h))
            y += viewport_size
        x += viewport_size
    return tiles


class BudgetTracker:
    """Tracks how many queries remain and raises early if the budget is spent."""

    def __init__(self, total: int = config.TOTAL_QUERY_BUDGET):
        self.total     = total
        self.used: int = 0

    @property
    def remaining(self) -> int:
        return self.total - self.used

    def consume(self, n: int = 1) -> None:
        if self.used + n > self.total:
            raise RuntimeError(
                f"Query budget exhausted "
                f"(tried to use {self.used + n} / {self.total})."
            )
        self.used += n

    def can_query(self, n: int = 1) -> bool:
        return self.used + n <= self.total

    def __repr__(self) -> str:
        return f"BudgetTracker({self.used}/{self.total} used)"


class ObservationPlanner:
    """
    Generates an ordered sequence of :class:`Viewport` objects that
    maximises map coverage within the query budget.

    Args:
        num_seeds:    Number of round seeds (default 5).
        map_width:    Full map width in cells.
        map_height:   Full map height in cells.
        total_budget: Total queries allowed for the round.
    """

    def __init__(
        self,
        num_seeds: int  = config.NUM_SEEDS,
        map_width: int  = config.MAP_WIDTH,
        map_height: int = config.MAP_HEIGHT,
        total_budget: int = config.TOTAL_QUERY_BUDGET,
    ):
        self.num_seeds    = num_seeds
        self.map_width    = map_width
        self.map_height   = map_height
        self.budget       = BudgetTracker(total_budget)
        self._tiles       = tile_map(map_width, map_height)
        self._phase1_done = False

    # ── Phase 1: full-coverage tiling ───────────────────────────────────────

    def phase1_viewports(self) -> list[Viewport]:
        """
        All viewports needed to fully tile the map for every seed.

        The order interleaves seeds so that if the run is interrupted we still
        have partial coverage of all seeds rather than full coverage of only
        the first few.
        """
        queries: list[Viewport] = []
        for tile_idx, (x, y, w, h) in enumerate(self._tiles):
            for seed_id in range(self.num_seeds):
                queries.append(Viewport(seed_id=seed_id, x=x, y=y, width=w, height=h))
        return queries

    def phase1_cost(self) -> int:
        return len(self._tiles) * self.num_seeds

    # ── Phase 2: uncertainty-driven refinement ───────────────────────────────

    def phase2_viewports(
        self,
        entropy_map: np.ndarray,
        seed_id: int,
    ) -> list[Viewport]:
        """
        Return up to ``remaining`` viewports centred on the highest-entropy
        regions for a given seed.

        Args:
            entropy_map: Per-cell mean entropy, shape ``(H, W)``.
            seed_id:     Which seed to refine.
        """
        available = self.budget.remaining
        if available <= 0:
            return []

        viewports: list[Viewport] = []
        seen_centres: set[tuple[int, int]] = set()

        # Smooth the entropy map to find the hottest region.
        flat_idx = np.argsort(entropy_map.ravel())[::-1]
        half = config.VIEWPORT_MAX_SIZE // 2

        for idx in flat_idx:
            if not self.budget.can_query():
                break
            cy, cx = divmod(int(idx), self.map_width)
            # Snap viewport so it stays inside the map.
            x = max(0, min(cx - half, self.map_width  - config.VIEWPORT_MAX_SIZE))
            y = max(0, min(cy - half, self.map_height - config.VIEWPORT_MAX_SIZE))
            centre = (x, y)
            if centre in seen_centres:
                continue
            seen_centres.add(centre)
            w = min(config.VIEWPORT_MAX_SIZE, self.map_width  - x)
            h = min(config.VIEWPORT_MAX_SIZE, self.map_height - y)
            viewports.append(Viewport(seed_id=seed_id, x=x, y=y, width=w, height=h))

            if len(viewports) >= available:
                break

        return viewports
