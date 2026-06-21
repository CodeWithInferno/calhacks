"""Generate synthetic demo rollouts matching the real G1 data schema."""

import os
import numpy as np
import pandas as pd


def generate_rollout(
    slope_angle_deg: float,
    friction: float,
    force_mag: float,
    force_body: int,
    seed: int,
    dt: float = 0.02,
    max_steps: int = 300,
    horizon: int = 25,
):
    rng = np.random.default_rng(seed)
    slope = np.deg2rad(slope_angle_deg)

    # Stability budget: lower -> easier to fall.
    stability_budget = 0.6 - 0.025 * slope_angle_deg - 0.0025 * force_mag - 0.1 * (1.0 - friction)
    stability_budget += rng.normal(0, 0.08)

    t = np.arange(max_steps) * dt
    timestep = np.arange(max_steps)

    # Base orientation and velocity.
    base_pos_x = np.cumsum(rng.normal(0.5, 0.05, size=max_steps) * dt)
    base_pos_y = rng.normal(0, 0.02, size=max_steps)
    base_pos_z = 0.8 - 0.0005 * force_mag * timestep / max_steps + rng.normal(0, 0.005, size=max_steps)

    base_quat_w = np.ones(max_steps) + rng.normal(0, 0.01, size=max_steps)
    base_quat_x = rng.normal(0, 0.03, size=max_steps)
    base_quat_y = np.full(max_steps, np.sin(slope / 2)) + rng.normal(0, 0.04, size=max_steps)
    base_quat_z = np.full(max_steps, np.cos(slope / 2)) + rng.normal(0, 0.01, size=max_steps)

    base_vel_x = 0.5 - 0.002 * force_mag + rng.normal(0, 0.05, size=max_steps)
    base_vel_y = rng.normal(0, 0.02, size=max_steps)
    base_vel_z = rng.normal(0, 0.02, size=max_steps)

    base_ang_vel_x = rng.normal(0, 0.05, size=max_steps)
    base_ang_vel_y = rng.normal(0, 0.06, size=max_steps) + 0.03 * (slope_angle_deg / 25.0)
    base_ang_vel_z = rng.normal(0, 0.03, size=max_steps)

    projected_gravity_x = rng.normal(0, 0.05, size=max_steps)
    projected_gravity_y = rng.normal(0, 0.05, size=max_steps)
    projected_gravity_z = -1.0 + rng.normal(0, 0.05, size=max_steps)

    cmd_vel_x = np.full(max_steps, 0.5)
    cmd_vel_y = np.zeros(max_steps)
    cmd_yaw_rate = np.zeros(max_steps)

    force_x = -force_mag * 0.7
    force_y = force_mag * 0.1
    force_z = -force_mag * 0.7
    force_app_x = base_pos_x + rng.normal(0, 0.05)
    force_app_y = base_pos_y + rng.normal(0, 0.05)
    force_app_z = base_pos_z + 0.3 + rng.normal(0, 0.05)

    # Approximate CoM: base position shifted slightly.
    robot_com_x = base_pos_x + rng.normal(0, 0.01, size=max_steps)
    robot_com_y = base_pos_y + rng.normal(0, 0.01, size=max_steps)
    robot_com_z = base_pos_z + 0.05 + rng.normal(0, 0.005, size=max_steps)

    # Combined system CoM shifts toward force application point.
    load_mass_ratio = force_mag / (force_mag + 500.0)
    system_com_x = (1 - load_mass_ratio) * robot_com_x + load_mass_ratio * force_app_x
    system_com_y = (1 - load_mass_ratio) * robot_com_y + load_mass_ratio * force_app_y
    system_com_z = (1 - load_mass_ratio) * robot_com_z + load_mass_ratio * force_app_z

    # Stability metric.
    stability = (
        stability_budget
        - 0.5 * np.abs(base_quat_x)
        - 0.3 * np.abs(base_quat_y - np.sin(slope / 2))
        - 0.2 * np.abs(base_ang_vel_y)
        - 0.02 * force_mag
        + 0.05 * base_vel_x
    )

    fall_candidates = np.where((stability < 0) & (timestep > 20))[0]
    fall_step = int(fall_candidates[0]) if len(fall_candidates) > 0 else None

    fall_label = np.zeros(max_steps, dtype=int)
    steps_to_fall = np.full(max_steps, -1, dtype=int)
    if fall_step is not None:
        mask = (timestep <= fall_step) & (fall_step <= timestep + horizon)
        fall_label[mask] = 1
        steps_to_fall[mask] = fall_step - timestep[mask]

    df = pd.DataFrame({
        "episode_id": seed,
        "time": t,
        "timestep": timestep,
        "slope_angle_deg": slope_angle_deg,
        "friction": friction,
        "base_pos_x": base_pos_x,
        "base_pos_y": base_pos_y,
        "base_pos_z": base_pos_z,
        "base_quat_w": base_quat_w,
        "base_quat_x": base_quat_x,
        "base_quat_y": base_quat_y,
        "base_quat_z": base_quat_z,
        "robot_com_x": robot_com_x,
        "robot_com_y": robot_com_y,
        "robot_com_z": robot_com_z,
        "system_com_x": system_com_x,
        "system_com_y": system_com_y,
        "system_com_z": system_com_z,
        "base_vel_x": base_vel_x,
        "base_vel_y": base_vel_y,
        "base_vel_z": base_vel_z,
        "base_ang_vel_x": base_ang_vel_x,
        "base_ang_vel_y": base_ang_vel_y,
        "base_ang_vel_z": base_ang_vel_z,
        "projected_gravity_x": projected_gravity_x,
        "projected_gravity_y": projected_gravity_y,
        "projected_gravity_z": projected_gravity_z,
        "cmd_vel_x": cmd_vel_x,
        "cmd_vel_y": cmd_vel_y,
        "cmd_yaw_rate": cmd_yaw_rate,
        "force_mag": force_mag,
        "force_x": force_x,
        "force_y": force_y,
        "force_z": force_z,
        "force_app_x": force_app_x,
        "force_app_y": force_app_y,
        "force_app_z": force_app_z,
        "fall_label": fall_label,
        "steps_to_fall": steps_to_fall,
    })

    return df


def main():
    os.makedirs("data", exist_ok=True)
    rng = np.random.default_rng(42)

    frames = []
    for i in range(500):
        slope = rng.uniform(0.0, 35.0)
        friction = rng.uniform(0.5, 1.0)
        force = rng.uniform(0.0, 200.0)
        body = rng.integers(0, 5)
        frames.append(generate_rollout(slope, friction, force, body, seed=i))

    df = pd.concat(frames, ignore_index=True)
    out_path = "data/demo_slope_load_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows from {df['episode_id'].nunique()} episodes")
    print(f"Saved to {out_path}")
    print(f"Fall rate: {df['fall_label'].mean():.3f}")


if __name__ == "__main__":
    main()
