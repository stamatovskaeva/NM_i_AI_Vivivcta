#!/usr/bin/env python
"""Analyze Round 15 vs Round 21 to understand the scoring difference."""

from api_client import AstarClient
import json

client = AstarClient()
r15_id = 'cc5442dd-bc5d-418b-911b-7eb960cb0390'
r21_id = 'b3a0be6b-b48b-419d-916a-b7a77fa58c4d'

print("=" * 80)
print("ROUND 15 vs ROUND 21 DETAILED COMPARISON")
print("=" * 80)

print("\nRound 15 (score=65.69, rank=185):")
print("-" * 80)
for seed_idx in range(5):
    try:
        analysis_r15 = client.get_analysis(r15_id, seed_idx)
        score = analysis_r15.get('score')
        print(f"  Seed {seed_idx}: score={score:.4f}")
    except Exception as e:
        print(f"  Seed {seed_idx}: Error - {e}")

print("\nRound 21 (score=24.06, rank=219):")
print("-" * 80)
for seed_idx in range(5):
    try:
        analysis_r21 = client.get_analysis(r21_id, seed_idx)
        score = analysis_r21.get('score')
        print(f"  Seed {seed_idx}: score={score:.4f}")
    except Exception as e:
        print(f"  Seed {seed_idx}: Error - {e}")

print("\n" + "=" * 80)
print("HYPOTHESIS: Why is Round 15 higher (65.69) vs Round 21 (24.06)?")
print("=" * 80)
print("""
Possible explanations:
1. Different scoring metric/formula between rounds
2. Different map difficulty (ground truth varies by round)
3. Different number of teams competing (affects ranking)
4. The 'score' in /my-rounds is different from per-seed scores
5. 65.69 aggregate vs 24.06 aggregate are on different scales
""")
