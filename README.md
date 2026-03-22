# NM_i_AI_Vivivcta
Repository for the world championship in Artificial Intelligence in Norway

## Post-round analysis

After a round reaches `scoring` or `completed`, run:

```powershell
cd .\Astar
python post_round_analysis.py
```

To analyze a specific round id:

```powershell
python post_round_analysis.py --round-id <round_id>
```

Outputs are saved under:

- `Astar/cache/post_round/round_<number>_<round_id>/analysis_summary.json`
- `Astar/cache/post_round/round_<number>_<round_id>/analysis_raw.json`

`analysis_summary.json` contains team score/rank metadata from `/my-rounds` plus
exact per-seed metrics computed from `/analysis/{round_id}/{seed_index}`:

- mean KL divergence
- mean cross-entropy
- top-1 accuracy
- dominant-class mismatch rate
- class-mass bias (`predicted - ground_truth`) for all 6 classes
