#!/usr/bin/env python
"""Fetch Round 14 data and compare to recent rounds."""

import json
from api_client import AstarClient

client = AstarClient()

# Get all rounds
print("Fetching all team rounds...")
rounds = client.get_my_rounds()

# Find Round 14
print("\nSearching for high-scoring rounds...\n")
print("-" * 90)

high_scores = []
for r in rounds:
    rid = r.get("id")
    num = r.get("round_number")
    score = r.get("round_score")
    rank = r.get("rank")
    queries = r.get("queries_used")
    seeds_submitted = r.get("seeds_submitted")
    
    if score is not None:
        high_scores.append((score, num, rid, rank, queries,seeds_submitted))
        print(f"Round {num:2d}: score={score:7.2f}, rank={rank:4d}, queries={queries}, seeds={seeds_submitted}, id={rid}")

print("\n" + "=" * 90)
print("Comparison by score (highest first):")
print("=" * 90)
for score, num, rid, rank, queries, seeds in sorted(high_scores, reverse=True)[:5]:
    print(f"Round {num:2d}: {score:7.2f} (rank {rank}, {queries} queries, {seeds} seeds submitted)")

# Focus on Round 14
round14 = [r for r in rounds if r.get("round_number") == 14]
if round14:
    r14 = round14[0]
    print(f"\n\nRound 14 Details:")
    print(f"  Score: {r14.get('round_score'):.2f}")
    print(f"  Rank: {r14.get('rank')}/{r14.get('total_teams')}")
    print(f"  Seed scores: {r14.get('seed_scores')}")
    print(f"  Queries used: {r14.get('queries_used')}/{r14.get('queries_max')}")
    print(f"  Submissions: {r14.get('seeds_submitted')}")
    print(f"  Round ID: {r14.get('id')}")


