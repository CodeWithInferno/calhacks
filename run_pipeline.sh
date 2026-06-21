#!/usr/bin/env bash
set -e

echo "==> Generating demo data"
python src/generate_demo_data.py

echo "==> Training world model"
python src/train_world_model.py

echo "==> Running inference"
python src/infer.py --input data/demo_slope_load_data.csv --output data/predictions.csv

echo "==> Evaluating"
python src/evaluate.py --input data/predictions.csv

echo "==> Done. Check data/predictions.csv and models/world_model/"
