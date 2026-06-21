# G1 Load & Slope Risk Copilot

A lightweight world model that predicts whether a Unitree G1 humanoid will fall within the next N steps while walking on a slope and carrying a virtual external load.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate demo synthetic data (until real CSV arrives)
python src/generate_demo_data.py

# Train the world model
python src/train_world_model.py

# Run inference
python src/infer.py --input data/demo_slope_load_data.csv --output data/predictions.csv

# Visualize one episode
python src/viz.py --input data/predictions.csv --episode 5 --output data/viz_episode_5.png
```

## Pipeline

1. **Data collection** (teammate): MuJoCo rollouts of G1 on random slopes + force vectors → `data/g1_slope_load_data.csv`.
2. **Training** (`src/train_world_model.py`): normalizes features, builds sliding windows, trains a small GRU/MLP classifier.
3. **Inference** (`src/infer.py`): appends `fall_risk` and `risk_flag` columns to a CSV.
4. **Visualization** (`src/viz.py`): plots risk curve vs. state for a selected episode.

## Config

Edit `src/config.yaml` to switch between `gru` and `mlp`, window size, hidden dim, etc.

## Expected CSV schema

| Column | Description |
|--------|-------------|
| `episode_id` | rollout identifier |
| `time` | simulation time (s) |
| `slope_angle_deg` | ground slope |
| `base_roll`, `base_pitch`, `base_pitch_rate` | base orientation / angular velocity |
| `base_vel_x`, `base_height` | forward velocity and CoM height |
| `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle` | simplified leg joints |
| `force_mag`, `force_x`, `force_z` | external force vector |
| `force_application_point` | integer body index |
| `fall_label` | 1 if fall occurs within prediction horizon |

When the real data schema differs, update `feature_cols` in `src/train_world_model.py` and `src/infer.py`.

## Nebius cloud training

The provided service account is scoped to an AI project (`aiproject-e00phab1pxk1ejwgmt`). `nebius compute instance create` requires a **folder/project** parent, not an AI project. To train on Nebius:

1. Log into [Nebius Console](https://console.nebius.com) and find the folder/project ID that contains the AI project.
2. Create a VM with GPU under that folder:
   ```bash
   nebius --profile hackathon compute instance create \
     --parent-id <folder-id> \
     --name g1-world-model \
     --platform gpu-h100-sxm \
     --preset 1gpu-h100-sxm \
     --zone eu-north1-c \
     --image-family ubuntu-22-04-lts \
     --ssh-user ubuntu \
     --ssh-public-key-file ~/.ssh/id_rsa.pub
   ```
3. SCP code + data and run `python src/train_world_model.py` on the VM.

## Team

- Pratham
- Hema
- Mahek
