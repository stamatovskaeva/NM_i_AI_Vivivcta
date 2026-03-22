"""
REST API client for Astar Island.

Matches the documented API at https://api.ainm.no/astar-island exactly.
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests

import config

logger = logging.getLogger(__name__)


# ── Data classes for API responses ───────────────────────────────────────────

@dataclass
class InitialSettlement:
    x: int
    y: int
    has_port: bool
    alive: bool


@dataclass
class InitialState:
    """Per-seed initial world state from GET /rounds/{id}."""
    grid: np.ndarray          # (H, W) int array, raw terrain codes
    settlements: list


@dataclass
class SimSettlement:
    """A settlement observed inside a viewport."""
    x: int
    y: int
    population: float
    food: float
    wealth: float
    defense: float
    has_port: bool
    alive: bool
    owner_id: int


@dataclass
class SimResult:
    """Full result of one POST /simulate call."""
    grid: np.ndarray          # (vp_h, vp_w) raw terrain codes
    settlements: list
    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int
    map_width: int
    map_height: int
    queries_used: int
    queries_max: int


class RoundInfo:
    """Parsed round metadata (from GET /rounds + GET /rounds/{id})."""

    def __init__(self, summary: dict, detail: dict):
        self.round_id: str       = summary["id"]
        self.round_number: int   = summary.get("round_number", 0)
        self.status: str         = summary.get("status", "unknown")
        self.map_width: int      = summary.get("map_width",  config.MAP_WIDTH)
        self.map_height: int     = summary.get("map_height", config.MAP_HEIGHT)
        self.seeds_count: int    = detail.get("seeds_count",  config.NUM_SEEDS)
        self.initial_states: list[InitialState] = [
            InitialState(
                grid=np.array(s["grid"], dtype=np.int32),
                settlements=[
                    InitialSettlement(
                        x=st["x"], y=st["y"],
                        has_port=st.get("has_port", False),
                        alive=st.get("alive", True),
                    )
                    for st in s.get("settlements", [])
                ],
            )
            for s in detail.get("initial_states", [])
        ]
        self.raw_summary = summary
        self.raw_detail  = detail

    def __repr__(self) -> str:
        return (
            f"RoundInfo(id={self.round_id!r}, "
            f"round={self.round_number}, status={self.status!r}, "
            f"map={self.map_width}×{self.map_height}, "
            f"seeds={self.seeds_count})"
        )


class BudgetInfo:
    """Parsed budget from GET /budget."""

    def __init__(self, data: dict):
        self.round_id: str     = data["round_id"]
        self.queries_used: int = data["queries_used"]
        self.queries_max: int  = data["queries_max"]
        self.active: bool      = data.get("active", True)

    @property
    def queries_remaining(self) -> int:
        return self.queries_max - self.queries_used

    def __repr__(self) -> str:
        return (
            f"BudgetInfo({self.queries_used}/{self.queries_max} used, "
            f"active={self.active})"
        )


class AstarClient:
    """
    Typed wrapper around the Astar Island REST API.

    Usage::

        client = AstarClient()
        info   = client.get_active_round()
        budget = client.get_budget()
        result = client.simulate(info.round_id, seed_index=0,
                                 vp_x=0, vp_y=0, vp_w=15, vp_h=15)
        client.submit_all_predictions(info.round_id, {0: tensor0, ...})
    """

    _RETRY_DELAYS = (1, 3, 8)   # seconds between retries on transient errors

    def __init__(
        self,
        token: str = config.AUTH_TOKEN,
        base_url: str = config.API_BASE,
    ):
        if not token:
            raise ValueError(
                "API token is empty. "
                "Set ASTAR_API_TOKEN in your .env file (see .env.example)."
            )
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    # ── private helpers ────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, **kwargs) -> dict | list:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_exc: Optional[Exception] = None
        for attempt, delay in enumerate((*self._RETRY_DELAYS, None), start=1):
            try:
                resp = self._session.request(method, url, timeout=30, **kwargs)
                if resp.status_code == 429:
                    # Rate-limited: wait the amount the server suggests, or 10 s.
                    wait = int(resp.headers.get("Retry-After", 10))
                    logger.warning("Rate-limited; sleeping %d s …", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                if delay is None:
                    break
                logger.warning(
                    "Request failed (attempt %d/%d): %s — retrying in %d s",
                    attempt,
                    len(self._RETRY_DELAYS) + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError(f"API request failed after retries: {last_exc}") from last_exc

    # ── public API ───────────────────────────────────────────────────────────

    def list_rounds(self) -> list[dict]:
        """GET /rounds — all rounds with status and timing."""
        return self._request("GET", "/rounds")  # type: ignore[return-value]

    def get_round(self, round_id: str) -> RoundInfo:
        """GET /rounds/{round_id} — full details including initial states."""
        rounds  = self.list_rounds()
        summary = next((r for r in rounds if r["id"] == round_id), {"id": round_id})
        detail  = self._request("GET", f"/rounds/{round_id}")
        return RoundInfo(summary, detail)  # type: ignore[arg-type]

    def get_active_round(self) -> RoundInfo:
        """
        Find the active round from GET /rounds and return its full details.
        Raises RuntimeError if no active round exists.
        """
        rounds = self.list_rounds()
        active = [r for r in rounds if r.get("status") == "active"]
        if not active:
            raise RuntimeError(
                "No active round found. Check app.ainm.no for the current round."
            )
        if len(active) > 1:
            logger.warning("Multiple active rounds; using the most recent one.")
            active.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        summary = active[0]
        detail  = self._request("GET", f"/rounds/{summary['id']}")
        info    = RoundInfo(summary, detail)  # type: ignore[arg-type]
        logger.info("Active round: %s", info)
        return info

    def get_budget(self) -> BudgetInfo:
        """GET /budget — remaining query budget for the active round."""
        data   = self._request("GET", "/budget")
        budget = BudgetInfo(data)  # type: ignore[arg-type]
        logger.info("Budget: %s", budget)
        return budget

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        vp_x: int,
        vp_y: int,
        vp_w: int,
        vp_h: int,
    ) -> SimResult:
        """
        POST /simulate — one stochastic simulation, costs one query.

        Grid values are raw terrain codes (0,1,2,3,4,5,10,11).
        Use ``config.TERRAIN_TO_CLASS`` to convert to prediction class indices.
        """
        if not (config.VIEWPORT_MIN_SIZE <= vp_w <= config.VIEWPORT_MAX_SIZE):
            raise ValueError(
                f"vp_w={vp_w} must be {config.VIEWPORT_MIN_SIZE}–{config.VIEWPORT_MAX_SIZE}"
            )
        if not (config.VIEWPORT_MIN_SIZE <= vp_h <= config.VIEWPORT_MAX_SIZE):
            raise ValueError(
                f"vp_h={vp_h} must be {config.VIEWPORT_MIN_SIZE}–{config.VIEWPORT_MAX_SIZE}"
            )
        payload = {
            "round_id":   round_id,
            "seed_index": seed_index,
            "viewport_x": vp_x,
            "viewport_y": vp_y,
            "viewport_w": vp_w,
            "viewport_h": vp_h,
        }
        data = self._request("POST", "/simulate", json=payload)
        grid = np.array(data["grid"], dtype=np.int32)  # type: ignore[index]
        settlements = [
            SimSettlement(
                x=s["x"], y=s["y"],
                population=float(s.get("population", 0)),
                food=float(s.get("food", 0)),
                wealth=float(s.get("wealth", 0)),
                defense=float(s.get("defense", 0)),
                has_port=bool(s.get("has_port", False)),
                alive=bool(s.get("alive", True)),
                owner_id=int(s.get("owner_id", -1)),
            )
            for s in data.get("settlements", [])  # type: ignore[union-attr]
        ]
        vp_conf = data.get("viewport", {})  # type: ignore[union-attr]
        result = SimResult(
            grid=grid,
            settlements=settlements,
            viewport_x=int(vp_conf.get("x", vp_x)),
            viewport_y=int(vp_conf.get("y", vp_y)),
            viewport_w=int(vp_conf.get("w", vp_w)),
            viewport_h=int(vp_conf.get("h", vp_h)),
            map_width=int(data.get("width",  config.MAP_WIDTH)),    # type: ignore[union-attr]
            map_height=int(data.get("height", config.MAP_HEIGHT)),  # type: ignore[union-attr]
            queries_used=int(data.get("queries_used", 0)),          # type: ignore[union-attr]
            queries_max=int(data.get("queries_max",   config.TOTAL_QUERY_BUDGET)),  # type: ignore[union-attr]
        )
        logger.info(
            "Query #%d/%d  seed=%d  vp=(%d,%d %d×%d)  settlements=%d",
            result.queries_used, result.queries_max, seed_index,
            result.viewport_x, result.viewport_y,
            result.viewport_w, result.viewport_h,
            len(settlements),
        )
        return result

    def submit_prediction(
        self,
        round_id: str,
        seed_index: int,
        tensor: np.ndarray,
    ) -> dict:
        """
        POST /submit — submit prediction for one seed.

        Re-submitting the same seed overwrites the previous prediction.
        """
        if tensor.ndim != 3 or tensor.shape[2] != config.NUM_TERRAIN_CLASSES:
            raise ValueError(
                f"Tensor for seed {seed_index} has wrong shape {tensor.shape}; "
                f"expected (H, W, {config.NUM_TERRAIN_CLASSES})."
            )
        sums = tensor.sum(axis=2)
        if not np.allclose(sums, 1.0, atol=0.01):
            raise ValueError(
                f"Seed {seed_index}: probs don't sum to 1 "
                f"(max deviation {np.abs(sums - 1.0).max():.6f})."
            )
        payload = {
            "round_id":   round_id,
            "seed_index": seed_index,
            "prediction": tensor.tolist(),
        }
        result = self._request("POST", "/submit", json=payload)
        logger.info("Seed %d submitted: %s", seed_index, result)
        return result  # type: ignore[return-value]

    def submit_all_predictions(
        self,
        round_id: str,
        predictions: dict[int, np.ndarray],
    ) -> list[dict]:
        """Submit all seeds by calling :meth:`submit_prediction` per seed."""
        return [
            self.submit_prediction(round_id, idx, predictions[idx])
            for idx in sorted(predictions)
        ]

    def get_my_rounds(self) -> list[dict]:
        """GET /my-rounds — team-specific rounds with scores/rank/budget."""
        return self._request("GET", "/my-rounds")  # type: ignore[return-value]

    def get_my_predictions(self, round_id: str) -> list[dict]:
        """GET /my-predictions/{round_id} — submitted predictions + confidence grids."""
        return self._request("GET", f"/my-predictions/{round_id}")  # type: ignore[return-value]

    def get_analysis(self, round_id: str, seed_index: int) -> dict:
        """
        GET /analysis/{round_id}/{seed_index} — post-round prediction vs ground truth.

        Requires round status to be ``scoring`` or ``completed``.
        """
        return self._request("GET", f"/analysis/{round_id}/{seed_index}")  # type: ignore[return-value]
