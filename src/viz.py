"""
Visualize fall-risk predictions for one or more episodes.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_episode(df, episode_id, save_path=None):
    ep = df[df["episode_id"] == episode_id].reset_index(drop=True)
    if len(ep) == 0:
        print(f"Episode {episode_id} not found")
        return

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(ep["time"], ep["fall_risk"], label="fall risk", color="red")
    axes[0].axhline(0.5, color="gray", linestyle="--", label="threshold")
    axes[0].set_ylabel("Risk")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[0].set_title(f"Episode {episode_id} — Fall Risk")

    axes[1].plot(ep["time"], ep["base_roll"], label="roll")
    axes[1].plot(ep["time"], ep["base_pitch"], label="pitch")
    axes[1].set_ylabel("rad")
    axes[1].legend()
    axes[1].set_title("Base Orientation")

    axes[2].plot(ep["time"], ep["base_vel_x"], label="vel x")
    axes[2].plot(ep["time"], ep["base_height"], label="height")
    axes[2].set_ylabel("m / (m/s)")
    axes[2].legend()
    axes[2].set_title("Base Velocity / Height")

    axes[3].plot(ep["time"], ep["slope_angle_deg"], label="slope")
    axes[3].plot(ep["time"], ep["force_mag"], label="force mag")
    axes[3].set_ylabel("deg / N")
    axes[3].set_xlabel("time (s)")
    axes[3].legend()
    axes[3].set_title("Slope & External Force")

    if "fall_label" in ep.columns:
        fall_times = ep[ep["fall_label"] == 1]["time"]
        if len(fall_times) > 0:
            for ax in axes:
                ax.axvspan(fall_times.min(), fall_times.max(), color="red", alpha=0.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/predictions.csv")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    plot_episode(df, args.episode, args.output)


if __name__ == "__main__":
    main()
