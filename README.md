# G1 Load & Slope Risk Copilot

A learned world model that predicts whether a Unitree G1 humanoid will fall within the next N steps while walking on a slope and carrying a virtual external load.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate demo synthetic data
python src/generate_demo_data.py

# Train the world model
python src/train_world_model.py

# Run inference
python src/infer.py --input data/demo_slope_load_data.csv --output data/predictions.csv

# Evaluate
python src/evaluate.py --input data/predictions.csv

# Visualize one episode
python src/viz.py --input data/predictions.csv --episode 5 --output data/viz_episode_5.png

# Plot training curves
python src/plot_training_log.py --log models/world_model/training_log.csv --output data/training_curves.png
```

## Pipeline

1. **Data collection** (teammate): MuJoCo rollouts of G1 on random slopes + force vectors → `data/g1_slope_load_data.csv`.
2. **Training** (`src/train_world_model.py`): normalizes features, builds sliding windows, trains a GRU/MLP classifier with checkpoints, logs, and early stopping.
3. **Inference** (`src/infer.py`): appends `fall_risk` and `risk_flag` columns to a CSV.
4. **Visualization** (`src/viz.py`): plots risk curve vs. state for a selected episode.

## Current model

- **Architecture:** 2-layer GRU, 320 hidden units, ~1M parameters.
- **Input:** sliding window of 10 timesteps of state + slope + force features.
- **Output:** probability of fall within prediction horizon.
- **Training:** AdamW, ReduceLROnPlateau, early stopping, checkpointing, CSV logging.

Edit `src/config.yaml` to change model size, window size, learning rate, etc.

## Data schema

See `DATA_SCHEMA.md` for the exact CSV columns.

To adapt to a different CSV, update `src/schema.yaml` with the exact column names and run training again.

## Nebius cloud training

See `NEBIUS_SETUP.md`.

## Team

- Pratham
- Hema
- Mahek
