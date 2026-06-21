# Imagined vs Actual Pipeline

| Step | Imagined Plan | What Actually Happened | Status |
|------|---------------|------------------------|--------|
| 1. Simulation data | Run G1 in MuJoCo with random slopes + external forces, log every timestep of state | Built `src/mujoco_collector/collector.py` using a heuristic balance/velocity controller after the Unitree ONNX policy proved unstable in this MJCF | ✅ Collected |
| 2. Data CSV | `data/g1_slope_load_data.csv` with episode_id, timestep, base/joint state, slope, force, fall_label | Generated `/home/hemad/calhacks/data/g1_mujoco_data.csv` on the VM: **367,859 rows / 8,192 episodes** (~44 rows/episode, ~64% fall-label rate) | ✅ Ready |
| 3. Training pipeline | Train supervised GRU/MLP world model on real physics data | Trained on VM with CUDA; episode-wise split avoids leakage. Current best model `models/mujoco_valloss`: **test AUC 0.9971 / test F1 0.9798 / test accuracy 0.9744** (independently verified). | ✅ Trained |
| 4. Cloud GPU | Use Nebius L40S for training | VM `195.242.29.248` with L40S used successfully | ✅ Done |
| 5. Model size | Small GRU (~50k params) for demo | Trained a 2-layer GRU with 128 hidden units, dropout 0.5, weight decay 0.01 (~223k params) on real MuJoCo data | ✅ Done |
| 6. Evaluation | Evaluate on held-out real episodes | Splits by episode before building windows; metrics are verified independently with `verify_model.py`. A normalization bug that inflated training-script metrics has been fixed. | ✅ Done |
| 7. Demo / viz | Live risk curve + decision layer in MuJoCo/Rerun | Static training report exists; demo/eval is user's responsibility | ⚠️ Pending |
| 8. Team integration | Teammate sends CSV, we train | Built our own collector; Hema's RL training log is not used | ✅ Unblocked |

## What changed

- The Unitree ONNX velocity policy was tested in MuJoCo with the provided PD gains and several actuator setups. It stays numerically stable but the robot collapses within ~1 s because the MJCF dynamics/contact properties differ from the IsaacLab USD model the policy was trained on.
- To unblock data collection, the collector uses a randomized heuristic balance/velocity controller. It still logs full per-timestep state, slopes, friction, external pushes, and fall labels matching `src/schema.yaml`.
- Dataset was expanded from **91,602 rows / 2,048 episodes** to **367,859 rows / 8,192 episodes**.
- The trained world model uses an episode-wise train/val/test split and selects the checkpoint by **validation loss**. A near-zero std bug in normalization (constant features got huge normalized values) was fixed so training-script metrics now match independent verification.

## Reproducible verification (independent test set)

Run from the repo root with the data available:

```bash
python verify_model.py --config configs/mujoco_valloss.yaml
```

Example output:

```
Model: models/mujoco_valloss
Episodes -> train: 6143 | val: 820 | test: 1229
Test windows: 42789 (fall rate 0.636)
Independent test metrics:
  auc: 0.9971
  ap: 0.9984
  accuracy: 0.9744
  precision: 0.9829
  recall: 0.9767
  f1: 0.9798
```

## Recommended checkpoint for demo

- **Best by val loss:** `models/mujoco_valloss/best_model.pt` — AUC 0.9971, F1 0.9798.
- **Last checkpoint:** `models/mujoco_valloss/last_checkpoint.pt` — usually comparable or slightly better; verify with the script above.

## Remaining work

1. **Demo / integration:** wire the chosen checkpoint into the final demo and generate live risk curves/evaluation visuals. `src/infer.py` can be pointed at `configs/mujoco_valloss.yaml` to score windows.
2. **Policy:** the Unitree ONNX velocity policy is still unstable in MuJoCo; if the demo needs learned control, retrain a policy in this MJCF or fine-tune the ONNX policy.
