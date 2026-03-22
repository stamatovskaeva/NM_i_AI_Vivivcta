#!/usr/bin/env python
"""Analyze Round 15 vs Round 21 predicted probability distributions."""

import json
import numpy as np
from api_client import AstarClient

client = AstarClient()

r15_id = 'cc5442dd-bc5d-418b-911b-7eb960cb0390'
r21_id = 'b3a0be6b-b48b-419d-916a-b7a77fa58c4d'

print("=" * 90)
print("CALIBRATION ANALYSIS: Round 15 vs Round 21")
print("=" * 90)

# Analyze seed 0 for both rounds
seed_idx = 0

print(f"\nFetching Round 15, Seed {seed_idx}...")
r15_analysis = client.get_analysis(r15_id, seed_idx)
r15_pred = np.array(r15_analysis['prediction'], dtype=np.float32)
r15_gt = np.array(r15_analysis['ground_truth'], dtype=np.float32)

print(f"Fetching Round 21, Seed {seed_idx}...")
r21_analysis = client.get_analysis(r21_id, seed_idx)
r21_pred = np.array(r21_analysis['prediction'], dtype=np.float32)
r21_gt = np.array(r21_analysis['ground_truth'], dtype=np.float32)

print("\n" + "=" * 90)
print(f"ROUND 15 (Score: {r15_analysis['score']:.4f})")
print("=" * 90)

# Compute statistics for R15
r15_pred_argmax = np.argmax(r15_pred, axis=-1)
r15_gt_argmax = np.argmax(r15_gt, axis=-1)
r15_pred_max = np.max(r15_pred, axis=-1)
r15_accuracy = np.mean(r15_pred_argmax == r15_gt_argmax)

print(f"Accuracy (dominant class match): {r15_accuracy:.4f} ({r15_accuracy * 100:.2f}%)")
print(f"Mean prediction confidence (max prob): {r15_pred_max.mean():.4f}")
print(f"Std prediction confidence: {r15_pred_max.std():.4f}")

# Check calibration: are high-confidence predictions actually correct?
confident_mask = r15_pred_max > 0.8
if confident_mask.sum() > 0:
    conf_accuracy = np.mean(r15_pred_argmax[confident_mask] == r15_gt_argmax[confident_mask])
    print(f"Accuracy when confidence > 0.8: {conf_accuracy:.4f} ({conf_accuracy * 100:.2f}%)")
    print(f"  → Overconfident" if conf_accuracy < r15_accuracy else f"  → Well-calibrated")

print("\n" + "=" * 90)
print(f"ROUND 21 (Score: {r21_analysis['score']:.4f})")
print("=" * 90)

# Compute statistics for R21
r21_pred_argmax = np.argmax(r21_pred, axis=-1)
r21_gt_argmax = np.argmax(r21_gt, axis=-1)
r21_pred_max = np.max(r21_pred, axis=-1)
r21_accuracy = np.mean(r21_pred_argmax == r21_gt_argmax)

print(f"Accuracy (dominant class match): {r21_accuracy:.4f} ({r21_accuracy * 100:.2f}%)")
print(f"Mean prediction confidence (max prob): {r21_pred_max.mean():.4f}")
print(f"Std prediction confidence: {r21_pred_max.std():.4f}")

confident_mask = r21_pred_max > 0.8
if confident_mask.sum() > 0:
    conf_accuracy = np.mean(r21_pred_argmax[confident_mask] == r21_gt_argmax[confident_mask])
    print(f"Accuracy when confidence > 0.8: {conf_accuracy:.4f} ({conf_accuracy * 100:.2f}%)")
    print(f"  → Overconfident" if conf_accuracy < r21_accuracy else f"  → Well-calibrated")

print("\n" + "=" * 90)
print("PROBABILITY DISTRIBUTION COMPARISON")
print("=" * 90)

# Compare entropy of predictions
r15_pred_entropy = -np.sum(r15_pred * np.log(np.maximum(r15_pred, 1e-10)), axis=-1)
r21_pred_entropy = -np.sum(r21_pred * np.log(np.maximum(r21_pred, 1e-10)), axis=-1)

print(f"\nRound 15 pred entropy: {r15_pred_entropy.mean():.4f} (std: {r15_pred_entropy.std():.4f})")
print(f"Round 21 pred entropy: {r21_pred_entropy.mean():.4f} (std: {r21_pred_entropy.std():.4f})")

if r21_pred_entropy.mean() < r15_pred_entropy.mean():
    print("→ R21 predicts MORE confidently (sharper distributions)")
else:
    print("→ R21 predicts LESS confidently (flatter distributions)")

# Check how many cells have max prob > 0.9
r15_highconf = np.mean(r15_pred_max > 0.9)
r21_highconf = np.mean(r21_pred_max > 0.9)
print(f"\nCells with max prob > 0.9:")
print(f"  Round 15: {r15_highconf * 100:.1f}%")
print(f"  Round 21: {r21_highconf * 100:.1f}%")

print("\n" + "=" * 90)
print("HYPOTHESIS")
print("=" * 90)
print("""
If R21 is:
- More confident (higher max prob) but less accurate (higher KL) → OVERCONFIDENT
- Less confident (lower max prob) but less accurate → BAD PRIORS
- Same confidence but less accurate → WRONG PREDICTIONS
""")

if r21_pred_max.mean() > r15_pred_max.mean():
    print("✗ R21 appears OVERCONFIDENT — predicts higher probabilities but has worse KL")
elif r21_pred_entropy.mean() > r15_pred_entropy.mean():
    print("✗ R21 appears UNDERCONFIDENT — predicts flatter distributions")
else:
    print("? R21 predictions are similar in confidence but score worse → WRONG CLASS PREDICTIONS")
