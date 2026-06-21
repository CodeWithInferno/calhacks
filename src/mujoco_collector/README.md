# MuJoCo G1 Data Collector

Self-contained MuJoCo rollout collector for the Unitree G1 humanoid. It loads
`unitree_mujoco`'s `g1_29dof.xml`, randomizes terrain / friction / pushes /
commands each episode, runs a heuristic balance-and-step controller, and writes
a CSV matching `/home/hemad/calhacks/src/schema.yaml` and `DATA_SCHEMA.md`.

## What it does

1. Loads the G1 MJCF from `/home/hemad/unitree_mujoco/unitree_robots/g1/g1_29dof.xml`.
2. Swaps the original torque-motor actuators for MuJoCo `position` actuators
   using the PD gains from `deploy.yaml`.
3. Randomizes every episode:
   - slope angle: uniform 0–30°
   - floor friction: uniform 0.5–1.0
   - external push on `pelvis` or `torso_link`: magnitude 0–80 N
   - velocity command from the training ranges
4. Runs a heuristic controller that balances pelvis pitch/roll and tracks
   velocity commands.
5. Logs per-timestep state, joints, actions, and external force.
6. Labels `fall_label=1` for the last 25 timesteps before a fall and writes
   `steps_to_fall`.
7. Runs episodes in parallel using 16 CPU cores.

## Why a heuristic controller instead of the Unitree ONNX policy?

The Unitree velocity policy (`policy.onnx`) was tested extensively in this
MuJoCo model. The MJCF dynamics and contact properties differ from the IsaacLab
USD model the policy was trained on, so the robot collapses within about one
second. To unblock real per-timestep data collection, the collector falls back
to a robust heuristic controller that still produces rich fall-risk data.

## Setup

Activate the conda environment and ensure dependencies are installed:

```bash
source /home/hemad/miniconda3/etc/profile.d/conda.sh
conda activate env_calhacks
pip install mujoco
```

`unitree_mujoco` is expected at `/home/hemad/unitree_mujoco`.

## Run

From `/home/hemad/calhacks`:

```bash
source /home/hemad/miniconda3/etc/profile.d/conda.sh
conda activate env_calhacks
python -m src.mujoco_collector.collector --episodes 2048 --workers 16
```

Or:

```bash
python src/mujoco_collector/run_collector.py --episodes 2048 --workers 16
```

Output is written to `/home/hemad/calhacks/data/g1_mujoco_data.csv`.

## Output schema

The CSV contains the columns required by `src/schema.yaml` plus a few useful
extras (`robot_com_*`, `system_com_*`, `force_body`).
