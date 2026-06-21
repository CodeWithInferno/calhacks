#!/usr/bin/env python3
"""Convenience entry point for the MuJoCo G1 data collector."""
import os
import sys

# Allow running from the repo root: python src/mujoco_collector/run_collector.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mujoco_collector.collector import run_collector

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect randomized Unitree G1 MuJoCo rollout data."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1024,
        help="Number of episodes to collect (default: 1024).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel worker processes (default: 16).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/hemad/calhacks/data/g1_mujoco_data.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    path, rows, elapsed = run_collector(
        n_episodes=args.episodes,
        n_workers=args.workers,
        output_path=args.output,
    )
    import pandas as pd

    eps = pd.read_csv(path)["episode_id"].nunique()
    print(f"Saved {path}")
    print(f"Rows: {rows} | Episodes: {eps} | Time: {elapsed:.1f}s")
