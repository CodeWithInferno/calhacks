#!/usr/bin/env bash
# Setup training environment on the Nebius VM.
set -e

source /home/hemad/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q env_calhacks; then
    conda create -y -n env_calhacks python=3.10
fi

conda activate env_calhacks

pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn pyyaml matplotlib imageio

echo "Setup complete. Activate with: conda activate env_calhacks"
