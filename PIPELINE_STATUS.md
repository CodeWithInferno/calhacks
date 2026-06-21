# Imagined vs Actual Pipeline

| Step | Imagined Plan | What Actually Happened | Status |
|------|---------------|------------------------|--------|
| 1. Simulation data | Run G1 in MuJoCo with random slopes + external forces, log every timestep of state | Built `src/mujoco_collector/collector.py` using a heuristic balance/velocity controller after the Unitree ONNX policy proved unstable in this MJCF | ✅ Collected |
| 2. Data CSV | `data/g1_slope_load_data.csv` with episode_id, timestep, base/joint state, slope, force, fall_label | Generated `/home/hemad/calhacks/data/g1_mujoco_data.csv` on the VM: **91,602 rows / 2,048 episodes** (~44 rows/episode, 54% fall-label rate) | ✅ Ready |
| 3. Training pipeline | Train supervised GRU/MLP world model on real physics data | Trained on VM with CUDA; episode-wise split avoids leakage. Best model `models/mujoco_valloss`: **test AUC 0.9953 / test F1 0.9724 / best val loss 0.1134** | ✅ Trained |
| 4. Cloud GPU | Use Nebius L40S for training | VM `195.242.29.248` with L40S used successfully | ✅ Done |
| 5. Model size | Small GRU (~50k params) for demo | Trained a 2-layer GRU with 320 hidden units (~1.2M params) on real MuJoCo data | ✅ Done |
| 6. Evaluation | Evaluate on held-out real episodes | Now splits by episode before building windows; metrics are credible | ✅ Done |
| 7. Demo / viz | Live risk curve + decision layer in MuJoCo/Rerun | Static training report exists; demo/eval is user's responsibility | ⚠️ Pending |
| 8. Team integration | Teammate sends CSV, we train | Built our own collector; Hema's RL training log is not used | ✅ Unblocked |

## What changed

- The Unitree ONNX velocity policy was tested in MuJoCo with the provided PD gains and several actuator setups. It stays numerically stable but the robot collapses within ~1 s because the MJCF dynamics/contact properties differ from the IsaacLab USD model the policy was trained on.
- To unblock data collection, the collector uses a randomized heuristic balance/velocity controller. It still logs full per-timestep state, slopes, friction, external pushes, and fall labels matching `src/schema.yaml`.
- The trained world model is in `models/mujoco_valloss/` on the VM and has been copied locally. It uses an episode-wise train/val/test split and selects the checkpoint by **validation loss** (instead of AUC) to reduce overfitting.

## Remaining work

1. **Push to GitHub:** the VM commit is ready but `origin` is an HTTPS URL and the VM has no credentials. Either configure HTTPS credentials, switch to an SSH deploy key, or push from the local repo.
2. **Demo / integration:** wire the trained model into the final demo and generate live risk curves/evaluation visuals.
