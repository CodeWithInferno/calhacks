"""Fair head-to-head comparison of our heuristic controller vs the Unitree ONNX policy.

Runs N episodes with each controller under identical randomized conditions and reports:
- fall rate
- median survival steps / time
- distance traveled (x-direction)
- final base height

The ONNX policy observation is reconstructed to match the IsaacLab training config
(base_ang_vel, projected_gravity, velocity commands, joint_pos_rel, joint_vel_rel,
last_action) with history length 5.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.spatial.transform import Rotation as R

from src.mujoco_collector.collector import (
    ACTION_SCALE,
    DEFAULT_JOINT_POS_POLICY,
    DEFAULT_QPOS_MOTOR,
    INV_MAP,
    JOINT_IDS_MAP,
    JOINT_NAMES,
    STEP_DT,
    EpisodeConfig,
    build_model,
    rotate_vec_by_quat_inv,
)

POLICY_ONNX = Path(
    "/home/hemad/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx"
)
HISTORY_LENGTH = 5
N_POLICY_OBS = 96  # 3 + 3 + 3 + 29 + 29 + 29


class OnnxPolicyController:
    """Unitree velocity-policy controller via ONNX Runtime."""

    def __init__(self, model: mujoco.MjModel, cfg: EpisodeConfig, sess: ort.InferenceSession):
        self.cfg = cfg
        self.sess = sess
        self.joint_ids = np.array([model.joint(n).id for n in JOINT_NAMES], dtype=np.int32)
        self.qpos_adr = np.array([model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.qvel_adr = np.array([model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)

        # History buffer: oldest at index 0, newest at index -1.
        self.history = np.zeros((HISTORY_LENGTH, N_POLICY_OBS), dtype=np.float32)
        self.last_action = np.zeros(29, dtype=np.float32)

    def _compute_obs(self, data: mujoco.MjData, pelvis_id: int) -> np.ndarray:
        base_quat = data.body(pelvis_id).xquat.copy()
        cvel = data.body(pelvis_id).cvel.copy()

        base_ang_vel = cvel[:3] * 0.2
        projected_gravity = rotate_vec_by_quat_inv(base_quat, np.array([0.0, 0.0, -1.0]))
        velocity_commands = np.array(
            [self.cfg.cmd_vel[0], self.cfg.cmd_vel[1], self.cfg.cmd_yaw_rate],
            dtype=np.float32,
        )

        joint_pos_motor = data.qpos[self.qpos_adr].copy()
        joint_vel_motor = data.qvel[self.qvel_adr].copy()
        joint_pos_policy = joint_pos_motor[INV_MAP]
        joint_vel_policy = joint_vel_motor[INV_MAP]

        joint_pos_rel = joint_pos_policy - DEFAULT_JOINT_POS_POLICY
        joint_vel_rel = joint_vel_policy * 0.05

        obs = np.concatenate(
            [base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel, joint_vel_rel, self.last_action]
        ).astype(np.float32)
        return obs

    def compute(self, data: mujoco.MjData, pelvis_id: int):
        obs = self._compute_obs(data, pelvis_id)
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = obs

        obs_in = self.history.reshape(1, -1)
        action = self.sess.run(None, {"obs": obs_in})[0][0]
        self.last_action = action.astype(np.float32)

        # JointPositionAction: target = default + action * scale.
        target_policy = DEFAULT_JOINT_POS_POLICY + action * ACTION_SCALE
        target_motor = target_policy[INV_MAP]
        return target_motor.astype(np.float32), self.last_action.copy()


def run_episode_controller(cfg: EpisodeConfig, controller_type: str, sess: ort.InferenceSession | None) -> pd.DataFrame:
    """Run one episode with the chosen controller and return per-step rows."""
    slope_rad = np.deg2rad(cfg.slope_deg)
    model = build_model(slope_rad, cfg.friction)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002
    n_substeps = int(round(STEP_DT / model.opt.timestep))

    pelvis_id = model.body("pelvis").id
    torso_id = model.body("torso_link").id
    force_body_id = pelvis_id if cfg.force_body == "pelvis" else torso_id

    joint_ids = np.array([model.joint(n).id for n in JOINT_NAMES], dtype=np.int32)
    qpos_adr = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32)

    rng = np.random.default_rng(cfg.seed + 1000)
    if controller_type == "heuristic":
        from src.mujoco_collector.collector import HeuristicController
        controller = HeuristicController(model, cfg, rng)
    elif controller_type == "onnx":
        controller = OnnxPolicyController(model, cfg, sess)
    else:
        raise ValueError(f"Unknown controller: {controller_type}")

    # Reset.
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, 0.78]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    if controller_type == "heuristic":
        data.qpos[qpos_adr] = controller.ref
    else:
        data.qpos[qpos_adr] = DEFAULT_QPOS_MOTOR
    mujoco.mj_forward(model, data)

    rows = []
    action_policy = np.zeros(29, dtype=np.float32)

    for t in range(cfg.max_steps):
        target_motor, action_policy = controller.compute(data, pelvis_id)
        data.ctrl[:] = target_motor
        data.xfrc_applied[force_body_id, :3] = cfg.force
        data.xfrc_applied[force_body_id, 3:] = 0.0

        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

        base_pos = data.body(pelvis_id).xpos.copy()
        base_quat = data.body(pelvis_id).xquat.copy()

        rows.append({
            "timestep": t,
            "time": (t + 1) * STEP_DT,
            "base_pos_x": float(base_pos[0]),
            "base_pos_y": float(base_pos[1]),
            "base_pos_z": float(base_pos[2]),
            "controller": controller_type,
            "episode_id": cfg.episode_id,
            "seed": cfg.seed,
            "slope_deg": cfg.slope_deg,
            "friction": cfg.friction,
            "cmd_vel_x": float(cfg.cmd_vel[0]),
            "cmd_vel_y": float(cfg.cmd_vel[1]),
            "cmd_yaw_rate": float(cfg.cmd_yaw_rate),
            "force_mag": float(np.linalg.norm(cfg.force)),
        })

        rmat = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]).as_matrix()
        pitch = float(np.arcsin(np.clip(-rmat[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(rmat[2, 1], rmat[2, 2]))
        fallen = (base_pos[2] < 0.42) or (abs(pitch) > 1.0) or (abs(roll) > 1.0)
        if fallen:
            break

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    episodes = df.groupby("episode_id")
    n_eps = df["episode_id"].nunique()
    fall_count = 0
    distances = []
    durations = []
    final_heights = []

    for _, ep_df in episodes:
        last = ep_df.iloc[-1]
        distances.append(float(last["base_pos_x"]))
        durations.append(float(last["time"]))
        final_heights.append(float(last["base_pos_z"]))
        # fell if episode did not reach max_steps (cheap proxy; consistent with collector)
        if len(ep_df) < 300:
            fall_count += 1

    return {
        "episodes": n_eps,
        "fall_rate": fall_count / n_eps if n_eps else 0.0,
        "median_distance_m": float(np.median(distances)),
        "mean_distance_m": float(np.mean(distances)),
        "median_duration_s": float(np.median(durations)),
        "mean_duration_s": float(np.mean(durations)),
        "median_final_height_m": float(np.median(final_heights)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output", type=str, default="/tmp/controller_compare.csv")
    parser.add_argument("--mild", action="store_true", help="Use mild conditions (low slope, no pushes)")
    args = parser.parse_args()

    if not POLICY_ONNX.exists():
        raise FileNotFoundError(f"ONNX policy not found at {POLICY_ONNX}")

    sess = ort.InferenceSession(str(POLICY_ONNX), providers=["CPUExecutionProvider"])

    results = []
    start = time.time()
    for i in range(args.runs):
        seed = 10_000 + i
        cfg = EpisodeConfig(episode_id=i, seed=seed, max_steps=args.max_steps)
        if args.mild:
            rng = np.random.default_rng(seed)
            cfg.slope_deg = float(rng.uniform(0.0, 5.0))
            cfg.friction = float(rng.uniform(0.8, 1.0))
            cfg.force = np.zeros(3, dtype=np.float32)
            cfg.cmd_vel = rng.uniform([0.0, 0.0], [0.5, 0.0]).astype(np.float32)
            cfg.cmd_yaw_rate = 0.0
        df_h = run_episode_controller(cfg, "heuristic", None)
        df_h["controller"] = "heuristic"
        results.append(df_h)

        # same conditions for ONNX
        df_o = run_episode_controller(cfg, "onnx", sess)
        df_o["controller"] = "onnx"
        results.append(df_o)

    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(args.output, index=False)

    heuristic_summary = summarize(combined[combined["controller"] == "heuristic"])
    onnx_summary = summarize(combined[combined["controller"] == "onnx"])

    print(f"\nComparison over {args.runs} matched episodes per controller")
    print(f"Output: {args.output}")
    print(f"Total time: {time.time() - start:.1f}s\n")

    print("Heuristic controller:")
    for k, v in heuristic_summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nUnitree ONNX policy controller:")
    for k, v in onnx_summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
