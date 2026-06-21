#!/usr/bin/env bash
# Run training on the Nebius VM with CUDA.
set -e

source /home/hemad/miniconda3/etc/profile.d/conda.sh
conda activate env_calhacks

cd /home/hemad/calhacks

CONFIG=${1:-configs/large.yaml}

echo "Training with config: $CONFIG"
python src/train_world_model.py --config "$CONFIG"
