"""
Run world model inference on a CSV of states.
Outputs risk probabilities and optional decision flags.
"""

import os
import yaml
import argparse
import numpy as np
import pandas as pd
import torch

from train_world_model import FallPredictorGRU, FallPredictorMLP


def load_model(output_dir, input_dim, window_size, cfg):
    model_type = cfg["model_type"]
    hidden_dim = cfg["hidden_dim"]

    if model_type == "mlp":
        model = FallPredictorMLP(input_dim * window_size, hidden_dim)
    elif model_type == "gru":
        model = FallPredictorGRU(input_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt"), map_location="cpu"))
    model.eval()
    return model


def run_inference(csv_path, output_path=None, risk_threshold=0.5):
    with open("src/config.yaml") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    window_size = cfg["window_size"]

    feature_cols = [
        "slope_angle_deg",
        "base_roll",
        "base_pitch",
        "base_pitch_rate",
        "base_vel_x",
        "base_height",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "force_mag",
        "force_x",
        "force_z",
        "force_application_point",
    ]

    df = pd.read_csv(csv_path)
    means = pd.read_csv(os.path.join(output_dir, "feature_means.csv"), index_col=0)["0"]
    stds = pd.read_csv(os.path.join(output_dir, "feature_stds.csv"), index_col=0)["0"]

    # Normalize.
    df[feature_cols] = (df[feature_cols] - means) / stds

    model = load_model(output_dir, len(feature_cols), window_size, cfg)

    predictions = []
    episodes = df["episode_id"].unique()

    with torch.no_grad():
        for ep in episodes:
            ep_df = df[df["episode_id"] == ep].reset_index(drop=True)
            if len(ep_df) < window_size:
                predictions.extend([np.nan] * len(ep_df))
                continue
            features = ep_df[feature_cols].values
            for i in range(len(ep_df)):
                if i < window_size:
                    predictions.append(np.nan)
                    continue
                window = features[i - window_size : i]
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                logit = model(x).item()
                prob = 1.0 / (1.0 + np.exp(-logit))
                predictions.append(prob)

    df["fall_risk"] = predictions
    df["risk_flag"] = (df["fall_risk"] > risk_threshold).astype(int)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="data/predictions.csv", help="Path to output CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Risk threshold")
    args = parser.parse_args()

    run_inference(args.input, args.output, args.threshold)


if __name__ == "__main__":
    main()
