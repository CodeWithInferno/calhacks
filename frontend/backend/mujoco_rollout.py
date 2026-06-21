"""Generate a real MuJoCo rollout using the heuristic controller."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Map MuJoCo 29-dof joints to the 24 actuated joints in the frontend URDF.
_URDF_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
]

try:
    from scipy.spatial.transform import Rotation as R
    from src.mujoco_collector.collector import (
        STEP_DT,
        EpisodeConfig,
        HeuristicController,
        JOINT_NAMES,
        build_model,
        rotate_vec_by_quat_inv,
    )

    _URDF_TO_MOTOR = {name: i for i, name in enumerate(JOINT_NAMES) if name in _URDF_JOINTS}
    MUJOCO_AVAILABLE = True
except Exception as exc:
    _URDF_TO_MOTOR = {}
    MUJOCO_AVAILABLE = False
    MUJOCO_ERROR = str(exc)


def generate_rollout(params: dict):
    """Run one MuJoCo episode with the heuristic controller.

    Returns (frames, df) where frames is the viewer schema and df is the
    DataFrame needed by risk_model.score_dataframe().
    """
    if not MUJOCO_AVAILABLE:
        raise RuntimeError(f"MuJoCo collector not available: {MUJOCO_ERROR}")

    import mujoco

    seed = int(params.get("seed", 42))
    max_steps = int(round(params.get("seconds", 8.0) / STEP_DT))
    slope_deg = float(params.get("incline_deg", 0.0))
    friction = float(params.get("friction", 1.0))
    speed = float(params.get("speed_mps", 1.0))

    cfg = EpisodeConfig(
        episode_id=0,
        seed=seed,
        max_steps=max_steps,
        slope_deg=slope_deg,
        friction=friction,
        cmd_vel=np.array([speed, 0.0], dtype=np.float32),
        cmd_yaw_rate=0.0,
        force=np.zeros(3, dtype=np.float32),
        force_body="pelvis",
    )

    slope_rad = np.deg2rad(cfg.slope_deg)
    model = build_model(slope_rad, cfg.friction)
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002
    n_substeps = int(round(STEP_DT / model.opt.timestep))

    pelvis_id = model.body("pelvis").id
    force_body_id = pelvis_id

    joint_ids = np.array([model.joint(n).id for n in JOINT_NAMES], dtype=np.int32)
    qpos_adr = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32)
    qvel_adr = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=np.int32)

    rng = np.random.default_rng(cfg.seed + 1000)
    controller = HeuristicController(model, cfg, rng)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, 0.55]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_adr] = controller.ref
    mujoco.mj_forward(model, data)

    frames = []
    rows = []
    for t in range(cfg.max_steps):
        target_motor, action_policy = controller.compute(data, pelvis_id)
        data.ctrl[:] = target_motor
        data.xfrc_applied[force_body_id, :3] = cfg.force
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

        base_pos = data.body(pelvis_id).xpos.copy()
        base_quat = data.body(pelvis_id).xquat.copy()
        cvel = data.body(pelvis_id).cvel.copy()
        projected_gravity = rotate_vec_by_quat_inv(base_quat, np.array([0.0, 0.0, -1.0]))
        joint_pos_motor = data.qpos[qpos_adr].copy()
        joint_vel_motor = data.qvel[qvel_adr].copy()

        time = (t + 1) * STEP_DT
        row = {
            "episode_id": 0,
            "timestep": t,
            "time": time,
            "slope_angle_deg": cfg.slope_deg,
            "friction": cfg.friction,
            "base_pos_x": float(base_pos[0]),
            "base_pos_y": float(base_pos[1]),
            "base_pos_z": float(base_pos[2]),
            "base_quat_w": float(base_quat[0]),
            "base_quat_x": float(base_quat[1]),
            "base_quat_y": float(base_quat[2]),
            "base_quat_z": float(base_quat[3]),
            "robot_com_x": float(base_pos[0]),
            "robot_com_y": float(base_pos[1]),
            "robot_com_z": float(base_pos[2]),
            "system_com_x": float(base_pos[0]),
            "system_com_y": float(base_pos[1]),
            "system_com_z": float(base_pos[2]),
            "base_vel_x": float(cvel[3]),
            "base_vel_y": float(cvel[4]),
            "base_vel_z": float(cvel[5]),
            "base_ang_vel_x": float(cvel[0]),
            "base_ang_vel_y": float(cvel[1]),
            "base_ang_vel_z": float(cvel[2]),
            "projected_gravity_x": float(projected_gravity[0]),
            "projected_gravity_y": float(projected_gravity[1]),
            "projected_gravity_z": float(projected_gravity[2]),
            "cmd_vel_x": float(cfg.cmd_vel[0]),
            "cmd_vel_y": float(cfg.cmd_vel[1]),
            "cmd_yaw_rate": float(cfg.cmd_yaw_rate),
            **{f"joint_pos_{i}": float(joint_pos_motor[i]) for i in range(len(JOINT_NAMES))},
            **{f"joint_vel_{i}": float(joint_vel_motor[i]) for i in range(len(JOINT_NAMES))},
            **{f"joint_default_{i}": float(controller.ref[i]) for i in range(len(JOINT_NAMES))},
            **{f"last_action_{i}": float(action_policy[i]) for i in range(len(JOINT_NAMES))},
            "force_mag": 0.0,
            "force_x": 0.0,
            "force_y": 0.0,
            "force_z": 0.0,
            "force_app_x": float(base_pos[0]),
            "force_app_y": float(base_pos[1]),
            "force_app_z": float(base_pos[2]),
        }
        rows.append(row)

        # Frontend frame schema.
        joints_out = {}
        for urdf_name, motor_idx in _URDF_TO_MOTOR.items():
            joints_out[urdf_name] = round(float(joint_pos_motor[motor_idx]), 5)

        frames.append({
            "t": round(time, 4),
            "joints": joints_out,
            "root": {
                "pos": [round(float(base_pos[0]), 5), round(float(base_pos[1]), 5), round(float(base_pos[2]), 5)],
                "quat": [round(float(base_quat[0]), 6), round(float(base_quat[1]), 6),
                         round(float(base_quat[2]), 6), round(float(base_quat[3]), 6)],
            },
            "objects": [],
        })

        rmat = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]).as_matrix()
        pitch = float(np.arcsin(np.clip(-rmat[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(rmat[2, 1], rmat[2, 2]))
        fallen = (base_pos[2] < 0.42) or (abs(pitch) > 1.0) or (abs(roll) > 1.0)
        if fallen:
            break

    return frames, pd.DataFrame(rows)
