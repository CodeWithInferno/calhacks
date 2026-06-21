"""
Convert teammate's raw simulation CSV into the training schema.

Edit the mapping below once the real CSV columns are known.
"""

import os
import argparse
import numpy as np
import pandas as pd


# TODO: replace with real column names from the simulation CSV.
COLUMN_MAP = {
    # "raw_col": "schema_col"
    "episode_id": "episode_id",
    "time": "time",
    "slope": "slope_angle_deg",
    "base_roll": "base_roll",
    "base_pitch": "base_pitch",
    "base_pitch_rate": "base_pitch_rate",
    "base_vel_x": "base_vel_x",
    "base_height": "base_height",
    "lh": "left_hip",
    "rh": "right_hip",
    "lk": "left_knee",
    "rk": "right_knee",
    "la": "left_ankle",
    "ra": "right_ankle",
    "force_mag": "force_mag",
    "force_x": "force_x",
    "force_z": "force_z",
    "force_point": "force_application_point",
    "fall_label": "fall_label",
}

REQUIRED = [
    "episode_id",
    "time",
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
    "fall_label",
]


def ingest(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df = df.rename(columns=COLUMN_MAP)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}")

    # Ensure integer types.
    df["episode_id"] = df["episode_id"].astype(int)
    df["force_application_point"] = df["force_application_point"].astype(int)
    df["fall_label"] = df["fall_label"].astype(int)

    df = df[REQUIRED]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Ingested {len(df)} rows -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/g1_slope_load_data.csv")
    args = parser.parse_args()
    ingest(args.input, args.output)


if __name__ == "__main__":
    main()
