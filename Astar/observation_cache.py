"""Observation query cache for Astar Island.

Stores each query as one JSON line for easy replay and analysis.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import config
from api_client import SimResult


class ObservationCache:
    """Append-only cache of query responses."""

    def __init__(self, path: Path = config.OBS_QUERIES_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        round_id: str,
        seed_index: int,
        requested_x: int,
        requested_y: int,
        requested_w: int,
        requested_h: int,
        result: SimResult,
    ) -> None:
        settlements = [asdict(s) for s in result.settlements]
        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "round_id": round_id,
            "seed_index": seed_index,
            "request": {
                "x": requested_x,
                "y": requested_y,
                "w": requested_w,
                "h": requested_h,
            },
            "response": {
                "viewport": {
                    "x": result.viewport_x,
                    "y": result.viewport_y,
                    "w": result.viewport_w,
                    "h": result.viewport_h,
                },
                "map_width": result.map_width,
                "map_height": result.map_height,
                "queries_used": result.queries_used,
                "queries_max": result.queries_max,
                "grid": result.grid.tolist(),
                "settlements": settlements,
            },
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
