#!/usr/bin/env python
"""Offline replay: test new config on Round 21 using stored count tensors."""

import json
import numpy as np
from pathlib import Path
from world_model import WorldModel
from api_client import AstarClient
import config

r21_id = 'b3a0be6b-b48b-419d-916a-b7a77fa58c4d'

# Load stored observation counts for Round 21
obs_path = Path("cache/observations.json")
if not obs_path.exists():
    print(f"Error: {obs_path} does not exist. Cannot replay without cached counts.")
    exit(1)

print("=" * 90)
print("OFFLINE REPLAY: Round 21 with NEW config")
print("=" * 90)
print(f"\nCurrent config (NEW):")
print(f"  PHASE2_WEIGHT_ENTROPY={config.PHASE2_WEIGHT_ENTROPY}")
print(f"  PHASE2_WEIGHT_COAST={config.PHASE2_WEIGHT_COAST}")
print(f"  PHASE2_WEIGHT_FOREST={config.PHASE2_WEIGHT_FOREST}")
print(f"  CORRECTION_WEIGHT_VOLATILITY={config.CORRECTION_WEIGHT_VOLATILITY}")
print(f"  CORRECTION_WEIGHT_COVERAGE={config.CORRECTION_WEIGHT_COVERAGE}")
print(f"  PRIOR_BLEND={config.PRIOR_BLEND}")

# Load stored observation COUNT tensors
with open(obs_path) as f:
    obs_data = json.load(f)

print(f"\nLoaded count tensors for {len(obs_data)} seeds")

# Construct WorldModel and inject the stored counts
print("\nReconstructing WorldModel with stored counts...")
wm = WorldModel()

# Inject counts for each seed
for seed_id_str, count_tensor in obs_data.items():
    seed_id = int(seed_id_str)
    counts_array = np.array(count_tensor, dtype=np.float32)
    
    # Inject directly into internal _counts storage
    wm._counts[seed_id] = counts_array
    
    # Mark cells as observed where counts > 0
    wm._observed[seed_id] = (counts_array.sum(axis=-1) > 0)
    
    print(f"  Seed {seed_id}: {wm._observed[seed_id].sum():5d} observed cells")

print("✓ WorldModel reconstructed with cached observations")

# Compute predictions with NEW config
print("\nComputing predictions with NEW config...")
new_predictions = wm.all_predictions()

# Fetch ground truth for comparison
client = AstarClient()
print("Fetching Round 21 ground truth from API...")

# Compare per-seed
print("\n" + "=" * 90)
print("PER-SEED COMPARISON: NEW vs GROUND TRUTH")
print("=" * 90)

total_score_old = 0.0
total_score_new = 0.0
seed_count = 0

for seed_idx in range(5):
    try:
        analysis = client.get_analysis(r21_id, seed_idx)
        old_score = analysis['score']
        old_pred = np.array(analysis['prediction'], dtype=np.float32)
        gt = np.array(analysis['ground_truth'], dtype=np.float32)
        
        # New prediction
        new_pred = new_predictions[seed_idx]
        
        # Compute KL divergence: entropy-weighted KL
        eps = 1e-10
        
        # Old KL
        kl_old_per_cell = np.sum(gt * (np.log(gt + eps) - np.log(old_pred + eps)), axis=-1)
        entropy_per_cell = -np.sum(gt * np.log(gt + eps), axis=-1)
        nonzero_entropy = entropy_per_cell > 1e-6
        if nonzero_entropy.sum() > 0:
            weighted_kl_old = np.sum(entropy_per_cell[nonzero_entropy] * kl_old_per_cell[nonzero_entropy]) / np.sum(entropy_per_cell[nonzero_entropy])
        else:
            weighted_kl_old = 0.0
        
        # New KL
        kl_new_per_cell = np.sum(gt * (np.log(gt + eps) - np.log(new_pred + eps)), axis=-1)
        if nonzero_entropy.sum() > 0:
            weighted_kl_new = np.sum(entropy_per_cell[nonzero_entropy] * kl_new_per_cell[nonzero_entropy]) / np.sum(entropy_per_cell[nonzero_entropy])
        else:
            weighted_kl_new = 0.0
        
        # Score formula: 100 * exp(-3 * weighted_kl)
        score_old = max(0, min(100, 100 * np.exp(-3 * weighted_kl_old)))
        score_new = max(0, min(100, 100 * np.exp(-3 * weighted_kl_new)))
        
        # Accuracy
        old_argmax = np.argmax(old_pred, axis=-1)
        new_argmax = np.argmax(new_pred, axis=-1)
        gt_argmax = np.argmax(gt, axis=-1)
        old_acc_check = np.mean(old_argmax == gt_argmax)
        new_acc = np.mean(new_argmax == gt_argmax)
        
        total_score_old += score_old
        total_score_new += score_new
        seed_count += 1
        
        improvement = score_new - score_old
        improvement_pct = (improvement / score_old * 100) if score_old > 0 else 0
        acc_improvement = (new_acc - old_acc_check) * 100
        
        print(f"\nSeed {seed_idx}:")
        print(f"  Old: score={score_old:6.2f}  acc={old_acc_check*100:5.1f}%  KL={weighted_kl_old:.4f}")
        print(f"  New: score={score_new:6.2f}  acc={new_acc*100:5.1f}%  KL={weighted_kl_new:.4f}")
        print(f"  Δ:   {improvement:+6.2f} ({improvement_pct:+5.1f}%)  acc {acc_improvement:+5.1f}pp")
        
    except Exception as e:
        print(f"Seed {seed_idx}: Error - {e}")
        import traceback
        traceback.print_exc()

if seed_count > 0:
    avg_old = total_score_old / seed_count
    avg_new = total_score_new / seed_count
    overall_improvement = avg_new - avg_old
    overall_pct = (overall_improvement / avg_old * 100) if avg_old > 0 else 0
    
    print("\n" + "=" * 90)
    print("ROUND AGGREGATE")
    print("=" * 90)
    print(f"Old config (Round 21):     {avg_old:6.2f}")
    print(f"New config (replayed):     {avg_new:6.2f}")
    print(f"Improvement:              {overall_improvement:+6.2f} ({overall_pct:+5.1f}%)")
    print("=" * 90)
    
    if avg_new > avg_old + 0.5:
        print(f"\n✅ NEW CONFIG IS BETTER — confidently proceed to next round!")
    elif avg_new > avg_old - 0.5:
        print(f"\n⚠️  MIXED RESULTS — config change is marginal, monitor vs baseline")
    else:
        print(f"\n❌ NEW CONFIG IS WORSE — consider reverting before next round!")

