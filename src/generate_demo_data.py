"""
Generate synthetic demo data for G1 slope + load world model.
"""

import os
import numpy as np
import pandas as pd


def generate_rollout(
    slope_angle_deg: float,
    force_mag: float,
    force_point: int,
    seed: int,
    dt: float = 0.02,
    max_steps: int = 300,
    fall_prediction_horizon: int = 25,
):
    rng = np.random.default_rng(seed)
    slope = np.deg2rad(slope_angle_deg)

    stability_budget = 0.55 - 0.022 * slope_angle_deg - 0.0025 * force_mag
    stability_budget += rng.normal(0, 0.08)

    t = np.arange(max_steps)

    base_pitch = slope + rng.normal(0, 0.06, size=max_steps)
    base_roll = rng.normal(0, 0.06, size=max_steps)
    base_pitch_rate = rng.normal(0, 0.06, size=max_steps) + 0.03 * (slope_angle_deg / 25.0)
    base_vel_x = 0.5 - 0.002 * force_mag + rng.normal(0, 0.05, size=max_steps)
    base_height = 0.8 - 0.0005 * force_mag * t / max_steps + rng.normal(0, 0.005, size=max_steps)

    phase = 2 * np.pi * 1.0 * t * dt
    left_hip = -0.1 + 0.2 * np.sin(phase)
    right_hip = -0.1 + 0.2 * np.sin(phase + np.pi)
    left_knee = 0.3 + 0.3 * np.sin(phase + np.pi / 2)
    right_knee = 0.3 + 0.3 * np.sin(phase + np.pi / 2 + np.pi)
    left_ankle = -0.2 + rng.normal(0, 0.02, size=max_steps)
    right_ankle = -0.2 + rng.normal(0, 0.02, size=max_steps)

    force_x = -force_mag * 0.7
    force_z = -force_mag * 0.7

    stability = (
        stability_budget
        - 0.5 * np.abs(base_roll)
        - 0.3 * np.abs(base_pitch - slope)
        - 0.1 * np.abs(base_pitch_rate)
        - 0.02 * force_mag
        + 0.05 * base_vel_x
    )

    # Trigger fall at first t > 20 where stability < 0.
    fall_candidates = np.where((stability < 0) & (t > 20))[0]
    fall_step = int(fall_candidates[0]) if len(fall_candidates) > 0 else None

    fall_label = np.zeros(max_steps, dtype=int)
    steps_to_fall = np.full(max_steps, -1, dtype=int)
    if fall_step is not None:
        mask = (t <= fall_step) & (fall_step <= t + fall_prediction_horizon)
        fall_label[mask] = 1
        steps_to_fall[mask] = fall_step - t[mask]

    df = pd.DataFrame({
        "time": t * dt,
        "episode_id": seed,
        "slope_angle_deg": slope_angle_deg,
        "base_roll": base_roll,
        "base_pitch": base_pitch,
        "base_pitch_rate": base_pitch_rate,
        "base_vel_x": base_vel_x,
        "base_height": base_height,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "left_knee": left_knee,
        "right_knee": right_knee,
        "left_ankle": left_ankle,
        "right_ankle": right_ankle,
        "force_mag": force_mag,
        "force_x": force_x,
        "force_z": force_z,
        "force_application_point": force_point,
        "fall_label": fall_label,
        "steps_to_fall": steps_to_fall,
    })

    return df


def main():
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    n_rollouts = 500
    rng = np.random.default_rng(42)

    frames = []
    for i in range(n_rollouts):
        slope = rng.uniform(0.0, 35.0)
        force = rng.uniform(0.0, 200.0)
        force_point = rng.choice([0, 1, 2, 3, 4])
        df = generate_rollout(slope_angle_deg=slope, force_mag=force, force_point=force_point, seed=i)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(out_dir, "demo_slope_load_data.csv")
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} timesteps from {n_rollouts} rollouts.")
    print(f"Saved to {out_path}")
    print(f"Fall rate: {df['fall_label'].mean():.3f}")


if __name__ == "__main__":
    main()
