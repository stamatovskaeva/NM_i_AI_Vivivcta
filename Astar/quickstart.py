"""
Astar Island quickstart script.

Implements the official flow:
1) Get active round
2) Get round details
3) Query simulator once
4) Submit uniform baseline for all seeds

Usage examples:
  python quickstart.py --token <JWT>
  python quickstart.py --simulate-only --token <JWT>
  python quickstart.py --cookie --token <JWT>

You can also set ASTAR_API_TOKEN in .env and omit --token.
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import requests
from dotenv import load_dotenv

BASE = "https://api.ainm.no"


def build_session(token: str, use_cookie: bool) -> requests.Session:
    session = requests.Session()
    if use_cookie:
        session.cookies.set("access_token", token)
    else:
        session.headers["Authorization"] = f"Bearer {token}"
    session.headers["Accept"] = "application/json"
    session.headers["Content-Type"] = "application/json"
    return session


def get_active_round(session: requests.Session) -> dict[str, Any]:
    rounds = session.get(f"{BASE}/astar-island/rounds", timeout=30)
    rounds.raise_for_status()
    data = rounds.json()
    active = next((r for r in data if r.get("status") == "active"), None)
    if active is None:
        raise RuntimeError("No active round found.")
    return active


def list_rounds(session: requests.Session) -> list[dict[str, Any]]:
    response = session.get(f"{BASE}/astar-island/rounds", timeout=30)
    response.raise_for_status()
    return response.json()


def get_round_by_id(session: requests.Session, round_id: str) -> dict[str, Any]:
    rounds = session.get(f"{BASE}/astar-island/rounds", timeout=30)
    rounds.raise_for_status()
    data = rounds.json()
    selected = next((r for r in data if r.get("id") == round_id), None)
    if selected is None:
        raise RuntimeError(f"Round id not found: {round_id}")
    return selected


def get_budget(session: requests.Session) -> dict[str, Any]:
    response = session.get(f"{BASE}/astar-island/budget", timeout=30)
    response.raise_for_status()
    return response.json()


def get_round_detail(session: requests.Session, round_id: str) -> dict[str, Any]:
    detail = session.get(f"{BASE}/astar-island/rounds/{round_id}", timeout=30)
    detail.raise_for_status()
    return detail.json()


def query_simulator_once(
    session: requests.Session,
    round_id: str,
    seed_index: int,
    viewport_x: int,
    viewport_y: int,
    viewport_w: int,
    viewport_h: int,
) -> dict[str, Any]:
    response = session.post(
        f"{BASE}/astar-island/simulate",
        json={
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def submit_uniform_baseline(
    session: requests.Session,
    round_id: str,
    width: int,
    height: int,
    seeds_count: int,
) -> None:
    for seed_idx in range(seeds_count):
        prediction = np.full((height, width, 6), 1 / 6, dtype=np.float32)
        response = session.post(
            f"{BASE}/astar-island/submit",
            json={
                "round_id": round_id,
                "seed_index": seed_idx,
                "prediction": prediction.tolist(),
            },
            timeout=30,
        )
        print(f"Submit seed {seed_idx}: {response.status_code}")
        if response.status_code >= 400:
            print(response.text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Astar Island quickstart")
    parser.add_argument("--token", default="", help="JWT access token")
    parser.add_argument("--cookie", action="store_true", help="Use cookie auth instead of Bearer header")
    parser.add_argument("--round-id", default="", help="Use a specific round id instead of auto-selecting active")
    parser.add_argument("--show-rounds", action="store_true", help="List rounds (id, number, status, timing) and exit")
    parser.add_argument("--budget-only", action="store_true", help="Print current team budget and exit")
    parser.add_argument("--simulate-only", action="store_true", help="Run steps 1-3 only, do not submit")
    parser.add_argument("--allow-submit", action="store_true", help="Required safety flag: explicitly allow submission")
    parser.add_argument("--seed-index", type=int, default=0, help="Seed index for sample simulation")
    parser.add_argument("--x", type=int, default=10, help="Viewport x")
    parser.add_argument("--y", type=int, default=5, help="Viewport y")
    parser.add_argument("--w", type=int, default=15, help="Viewport width")
    parser.add_argument("--h", type=int, default=15, help="Viewport height")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    token = args.token or os.getenv("ASTAR_API_TOKEN", "")
    if not token:
        print("Missing token. Pass --token or set ASTAR_API_TOKEN in .env")
        return 1

    session = build_session(token, use_cookie=args.cookie)

    if args.show_rounds:
        rounds = list_rounds(session)
        rounds_sorted = sorted(rounds, key=lambda r: r.get("round_number", 0), reverse=True)
        for r in rounds_sorted:
            print(
                f"round={r.get('round_number')}  id={r.get('id')}  "
                f"status={r.get('status')}  started_at={r.get('started_at')}  closes_at={r.get('closes_at')}"
            )
        return 0

    if args.budget_only:
        budget = get_budget(session)
        print(
            "Budget: "
            f"round_id={budget.get('round_id')}  "
            f"queries={budget.get('queries_used')}/{budget.get('queries_max')}  "
            f"active={budget.get('active')}"
        )
        return 0

    if args.round_id:
        selected = get_round_by_id(session, args.round_id)
    else:
        selected = get_active_round(session)

    round_id = selected["id"]
    print(
        f"Selected round: {selected.get('round_number')}  "
        f"id={round_id}  status={selected.get('status')}"
    )

    detail = get_round_detail(session, round_id)
    width = detail["map_width"]
    height = detail["map_height"]
    seeds_count = detail["seeds_count"]
    print(f"Round details: {width}x{height}, seeds={seeds_count}")

    for index, state in enumerate(detail.get("initial_states", [])):
        settlements = state.get("settlements", [])
        print(f"Seed {index}: initial settlements={len(settlements)}")

    sim = query_simulator_once(
        session=session,
        round_id=round_id,
        seed_index=args.seed_index,
        viewport_x=args.x,
        viewport_y=args.y,
        viewport_w=args.w,
        viewport_h=args.h,
    )
    vp = sim.get("viewport", {})
    print(
        f"Sim query ok: viewport=({vp.get('x')},{vp.get('y')},{vp.get('w')}x{vp.get('h')}), "
        f"queries={sim.get('queries_used')}/{sim.get('queries_max')}"
    )
    print(f"Visible settlements in viewport: {len(sim.get('settlements', []))}")

    if args.simulate_only:
        return 0

    if not args.allow_submit:
        print("Submission blocked. Pass --allow-submit to explicitly consent.")
        return 1

    submit_uniform_baseline(
        session=session,
        round_id=round_id,
        width=width,
        height=height,
        seeds_count=seeds_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
