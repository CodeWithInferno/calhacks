# Imagined vs Actual Pipeline

| Step | Imagined Plan | What Actually Happened | Status |
|------|---------------|------------------------|--------|
| 1. Simulation data | Run G1 in MuJoCo with random slopes + external forces, log every timestep of state | No real MuJoCo data collected. Hema trained an RL walking policy in Isaac Lab instead. | ❌ Missing |
| 2. Data CSV | `data/g1_slope_load_data.csv` with episode_id, timestep, base/joint state, slope, force, fall_label | Got `results/g1_wildfire_terrain/g1_fall_stats.csv` — aggregate RL training metrics, not per-timestep state | ⚠️ Wrong format |
| 3. Training pipeline | Train supervised GRU/MLP world model on real physics data | Built working pipeline, but trained only on synthetic demo data | ⚠️ Demo only |
| 4. Cloud GPU | Use Nebius H100/L40S for training | Nebius CLI blocked: service account scoped to AI project, cannot create compute instances without folder/project ID | ❌ Blocked |
| 5. Model size | Small GRU (~50k params) for demo | Added benchmark configs up to XL (~8M params), but only ran small on local MPS | ⚠️ Configs ready, not benchmarked |
| 6. Evaluation | Evaluate on held-out real episodes | Evaluation is on synthetic data with same-episode leakage; metrics are optimistic | ❌ Not valid |
| 7. Demo / viz | Live risk curve + decision layer in MuJoCo/Rerun | Static matplotlib plots from synthetic data | ⚠️ Partial |
| 8. Team integration | Teammate sends CSV, we train | Miscommunication: teammate sent RL policy training log, not world-model dataset | ❌ Needs fix |

## What is blocking real progress

1. **No per-timestep rollout data.** This is the only thing that actually matters.
2. **No running G1 simulation** we can instrument to log states and forces.

## Two ways to unblock

### Option A: Use Hema's trained RL policy to collect data
- Load `model_2300.pt` in Isaac Lab or MuJoCo.
- Run rollouts on slope/rough terrain.
- Log every timestep: base state, joint state, terrain info, contact forces, fall flag.
- Save as CSV matching `DATA_SCHEMA.md`.

### Option B: Build a MuJoCo data collector ourselves
- Clone `unitree_mujoco` + G1 robot assets.
- Load the Unitree ONNX velocity policy.
- Modify scene for slopes and apply external forces.
- Log states and fall labels.

Both require the same thing: a real physics environment producing per-timestep robot state.
