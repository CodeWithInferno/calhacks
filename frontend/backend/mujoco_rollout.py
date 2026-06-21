"""Generate a real MuJoCo rollout using the heuristic or safe MPC controller."""
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
    import mujoco
    from scipy.spatial.transform import Rotation as R
    from src.mujoco_collector.collector import (
        STEP_DT,
        ACTION_SCALE,
        DEFAULT_JOINT_POS_POLICY,
        DEFAULT_QPOS_MOTOR,
        EpisodeConfig,
        HeuristicController,
        JOINT_NAMES,
        build_model,
        rotate_vec_by_quat_inv,
    )
    from risk_model import _FEATURE_COLS

    _URDF_TO_MOTOR = {name: i for i, name in enumerate(JOINT_NAMES) if name in _URDF_JOINTS}
    MUJOCO_AVAILABLE = True
    MUJOCO_ERROR = ""
except Exception as exc:
    _URDF_TO_MOTOR = {}
    MUJOCO_AVAILABLE = False
    MUJOCO_ERROR = str(exc)


def _state_to_features(
    data: "mujoco.MjData",
    model: "mujoco.MjModel",
    cfg: EpisodeConfig,
    pelvis_id: int,
    force_body_id: int,
    joint_ids: np.ndarray,
    qpos_adr: np.ndarray,
    qvel_adr: np.ndarray,
    action_policy: np.ndarray,
) -> np.ndarray:
    base_pos = data.body(pelvis_id).xpos.copy()
    base_quat = data.body(pelvis_id).xquat.copy()
    cvel = data.body(pelvis_id).cvel.copy()
    projected_gravity = rotate_vec_by_quat_inv(base_quat, np.array([0.0, 0.0, -1.0]))
    joint_pos_motor = data.qpos[qpos_adr].copy()
    joint_vel_motor = data.qvel[qvel_adr].copy()
    force_app = data.body(force_body_id).xpos.copy()

    values = [
        cfg.slope_deg, cfg.friction,
        float(base_pos[0]), float(base_pos[1]), float(base_pos[2]),
        float(base_quat[0]), float(base_quat[1]), float(base_quat[2]), float(base_quat[3]),
        float(base_pos[0]), float(base_pos[1]), float(base_pos[2]),
        float(base_pos[0]), float(base_pos[1]), float(base_pos[2]),
        float(cvel[3]), float(cvel[4]), float(cvel[5]),
        float(cvel[0]), float(cvel[1]), float(cvel[2]),
        float(projected_gravity[0]), float(projected_gravity[1]), float(projected_gravity[2]),
        float(cfg.cmd_vel[0]), float(cfg.cmd_vel[1]), float(cfg.cmd_yaw_rate),
    ]
    for i in range(len(JOINT_NAMES)):
        values.extend([
            float(joint_pos_motor[i]),
            float(joint_vel_motor[i]),
            float(DEFAULT_QPOS_MOTOR[i]),
            float(action_policy[i]),
        ])
    values.extend([
        float(np.linalg.norm(cfg.force)),
        float(cfg.force[0]), float(cfg.force[1]), float(cfg.force[2]),
        float(force_app[0]), float(force_app[1]), float(force_app[2]),
    ])
    return np.array(values, dtype=np.float32)


def run_episode_controller(
    cfg: EpisodeConfig,
    controller_type: str = "safe",
) -> tuple[list[dict], pd.DataFrame]:
    """Run one MuJoCo episode and return frontend frames + a feature DataFrame."""
    if not MUJOCO_AVAILABLE:
        raise RuntimeError(f"MuJoCo collector not available: {MUJOCO_ERROR}")

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
    qvel_adr = np.array([model.jnt_dofadr[jid] for jid in joint_ids], dtype=np.int32)

    rng = np.random.default_rng(cfg.seed + 1000)
    if controller_type == "heuristic":
        controller = HeuristicController(model, cfg, rng)
    elif controller_type == "safe":
        from safe_controller import SafeController
        controller = SafeController(model, cfg, rng, enable_mpc=True)
    elif controller_type == "reference":
        from src.mujoco_collector.reference_controller import ReferenceWalkController
        controller = ReferenceWalkController(model, cfg, rng)
    else:
        raise ValueError(f"Unknown controller: {controller_type}")

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

        time = (t + 1) * STEP_DT
        feat = _state_to_features(data, model, cfg, pelvis_id, force_body_id, joint_ids, qpos_adr, qvel_adr, action_policy)
        row = {"episode_id": 0, "timestep": t, "time": time}
        row.update({name: float(val) for name, val in zip(_FEATURE_COLS, feat)})
        rows.append(row)

        joint_pos_motor = data.qpos[qpos_adr].copy()
        joints_out = {name: round(float(joint_pos_motor[idx]), 5) for name, idx in _URDF_TO_MOTOR.items()}
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


def generate_rollout(params: dict, controller_type: str = "safe"):
    """Run one MuJoCo episode and return (frames, df, cfg)."""
    if not MUJOCO_AVAILABLE:
        raise RuntimeError(f"MuJoCo collector not available: {MUJOCO_ERROR}")

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

    frames, df = run_episode_controller(cfg, controller_type)
    return frames, df, cfg
