"""MuJoCo data collector for Unitree G1.

This script loads the Unitree G1 29-DoF MJCF from ``unitree_mujoco``,
randomizes slope/friction/external forces/velocity commands every episode,
and runs a simple heuristic balance-and-step controller.  The original
Unitree ONNX velocity policy was attempted but is not directly stable under
this MuJoCo model (the MJCF dynamics/contact properties differ from the
IsaacLab USD model the policy was trained on), so we fall back to a robust
heuristic controller that still produces rich fall-risk data.

The output CSV follows ``/home/hemad/calhacks/src/schema.yaml`` and
``/home/hemad/calhacks/DATA_SCHEMA.md``.
"""
from __future__ import annotations

import os
import re
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
UNITREE_MUJOCO_DIR = Path("/home/hemad/unitree_mujoco")
G1_XML = UNITREE_MUJOCO_DIR / "unitree_robots" / "g1" / "g1_29dof.xml"
SCENE_XML = UNITREE_MUJOCO_DIR / "unitree_robots" / "g1" / "scene_29dof.xml"
POLICY_ONNX = Path(
    "/home/hemad/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx"
)

STEP_DT = 0.02
CTRL_FREQ = 1.0 / STEP_DT
N_ACT = 29
N_HISTORY = 5  # kept for documentation; heuristic does not need history

# Default joint positions in *policy action order* (from deploy.yaml).
DEFAULT_JOINT_POS_POLICY = np.array(
    [
        -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.3, 0.3, 0.3, 0.3, -0.2, -0.2, 0.25, -0.25,
        0.0, 0.0, 0.0, 0.0, 0.97, 0.97, 0.15, -0.15,
        0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)

# joint_ids_map[i] = motor index that policy action i drives.
JOINT_IDS_MAP = np.array(
    [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
        4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26,
        20, 27, 21, 28,
    ],
    dtype=np.int32,
)
INV_MAP = np.empty(29, dtype=np.int32)
INV_MAP[JOINT_IDS_MAP] = np.arange(29)

ACTION_SCALE = np.full(29, 0.25, dtype=np.float32)
DEFAULT_QPOS_MOTOR = DEFAULT_JOINT_POS_POLICY[INV_MAP]

# Stiffness / damping in *motor order* from deploy.yaml.
STIFFNESS = np.array(
    [
        100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
        100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
        200.0, 200.0, 200.0,
        40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    ],
    dtype=np.float32,
)
DAMPING = np.array(
    [
        2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
        5.0, 5.0, 5.0,
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
    ],
    dtype=np.float32,
)

JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


# --------------------------------------------------------------------------- #
# Episode configuration
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeConfig:
    episode_id: int
    seed: int
    max_steps: int = 300
    slope_deg: float = 0.0
    friction: float = 1.0
    cmd_vel: np.ndarray = None
    cmd_yaw_rate: float = 0.0
    force_body: str = "pelvis"
    force: np.ndarray = None
    randomize: bool = True

    def __post_init__(self):
        if self.cmd_vel is None:
            self.cmd_vel = np.zeros(2, dtype=np.float32)
        if self.force is None:
            self.force = np.zeros(3, dtype=np.float32)
        if not self.randomize:
            return
        rng = np.random.default_rng(self.seed)
        self.slope_deg = float(rng.uniform(0.0, 30.0))
        self.friction = float(rng.uniform(0.5, 1.0))
        self.cmd_vel = rng.uniform([-0.5, -0.3], [1.0, 0.3]).astype(np.float32)
        self.cmd_yaw_rate = float(rng.uniform(-0.2, 0.2))
        self.force_body = rng.choice(["pelvis", "torso_link"])
        force_mag = float(rng.uniform(0.0, 80.0))
        force_dir = rng.normal(size=3).astype(np.float32)
        force_dir /= np.linalg.norm(force_dir) + 1e-8
        self.force = (force_dir * force_mag).astype(np.float32)


# --------------------------------------------------------------------------- #
# MuJoCo model builder
# --------------------------------------------------------------------------- #
def build_model(slope_rad: float, friction: float) -> mujoco.MjModel:
    """Load G1 MJCF, swap in position actuators and a sloped floor, compile."""
    g1_xml = G1_XML.read_text()
    # Remove the original torque-motor actuators; we will add position actuators.
    g1_xml = re.sub(r"<actuator>.*?</actuator>\s*", "", g1_xml, flags=re.DOTALL)

    act_lines = ["  <actuator>"]
    for i, jn in enumerate(JOINT_NAMES):
        act_lines.append(
            f'    <position name="{jn}_pos" joint="{jn}" '
            f'kp="{STIFFNESS[i]:.1f}" kv="{DAMPING[i]:.1f}" '
            f'ctrlrange="-1000 1000"/>'
        )
    act_lines.append("  </actuator>")
    g1_xml = g1_xml.replace("</mujoco>", "\n".join(act_lines) + "\n</mujoco>")

    # Add scene visuals / lights / floor.
    scene_xml = SCENE_XML.read_text()
    scene_extra = scene_xml[scene_xml.find("<statistic") : scene_xml.find("</mujoco>")]
    full_xml = g1_xml.replace("</mujoco>", scene_extra + "\n</mujoco>")

    # Sloped floor (rotated box) with randomized friction.
    cy = np.cos(slope_rad * 0.5)
    sy = np.sin(slope_rad * 0.5)
    full_xml = full_xml.replace(
        '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>',
        f'<geom name="floor" size="50 50 0.05" type="box" pos="0 0 -0.1" '
        f'quat="{cy} 0 {sy} 0" friction="{friction:.4f} 0.005 0.0001" '
        f'rgba="0.5 0.5 0.55 1"/>',
    )

    if "<option" not in full_xml:
        full_xml = full_xml.replace(
            "<worldbody>",
            '<option timestep="0.002" iterations="50" solver="Newton"/>\n  <worldbody>',
        )

    tmp_xml = G1_XML.parent / f"g1_29dof_collector_{uuid.uuid4().hex[:8]}.xml"
    try:
        tmp_xml.write_text(full_xml)
        model = mujoco.MjModel.from_xml_path(str(tmp_xml))
    finally:
        if tmp_xml.exists():
            tmp_xml.unlink()
    return model


# --------------------------------------------------------------------------- #
# Heuristic controller
# --------------------------------------------------------------------------- #
class HeuristicController:
    """Simple balance + velocity-tracking controller.

    Returns *motor-order* target positions and a corresponding
    *policy-order* raw action vector (deviation from default / scale).
    """

    def __init__(self, model: mujoco.MjModel, cfg: EpisodeConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.joint_ids = np.array([model.joint(n).id for n in JOINT_NAMES], dtype=np.int32)
        self.jnt_range = np.stack([model.jnt_range[jid] for jid in self.joint_ids])

        # Randomized low-COM reference pose for stability.
        self.ref = DEFAULT_QPOS_MOTOR.copy()
        knee_flex = float(rng.uniform(0.5, 0.9))
        hip_flex = float(rng.uniform(0.25, 0.5))
        ankle_flex = float(rng.uniform(0.15, 0.4))
        self.ref[0] = -hip_flex
        self.ref[6] = -hip_flex
        self.ref[3] = knee_flex
        self.ref[9] = knee_flex
        self.ref[4] = -ankle_flex
        self.ref[10] = -ankle_flex

        self.Kp_pitch = float(rng.uniform(2.0, 5.0))
        self.Kd_pitch = float(rng.uniform(0.3, 0.8))
        self.Kp_roll = float(rng.uniform(2.0, 5.0))
        self.Kd_roll = float(rng.uniform(0.3, 0.8))

        self.action_noise = np.zeros(N_ACT, dtype=np.float32)

    def compute(
        self,
        data: mujoco.MjData,
        pelvis_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (target_motor positions, action_policy)."""
        target = self.ref.copy()

        # Velocity-command feedforward.
        vx, vy = self.cfg.cmd_vel
        yaw = self.cfg.cmd_yaw_rate
        target[0] += -0.12 * vx
        target[6] += -0.12 * vx
        target[4] += 0.06 * vx
        target[10] += 0.06 * vx
        target[1] += 0.12 * vy
        target[7] += -0.12 * vy
        target[5] += 0.05 * vy
        target[11] += -0.05 * vy
        target[2] += 0.12 * yaw
        target[8] += -0.12 * yaw

        # Balance feedback from pelvis pitch/roll.
        quat = data.body(pelvis_id).xquat
        rmat = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        pitch = float(np.arcsin(np.clip(-rmat[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(rmat[2, 1], rmat[2, 2]))
        dpitch = float(data.body(pelvis_id).cvel[0])
        droll = float(data.body(pelvis_id).cvel[1])

        tau_pitch = self.Kp_pitch * pitch + self.Kd_pitch * dpitch
        tau_roll = self.Kp_roll * roll + self.Kd_roll * droll

        target[0] += tau_pitch
        target[6] += tau_pitch
        target[4] += -1.2 * tau_pitch
        target[10] += -1.2 * tau_pitch
        target[1] += -tau_roll
        target[7] += tau_roll
        target[5] += 0.5 * tau_roll
        target[11] += -0.5 * tau_roll

        # Small exploratory action noise (Ornstein-Uhlenbeck-ish).
        self.action_noise = 0.95 * self.action_noise + 0.05 * self.rng.normal(size=N_ACT).astype(np.float32)
        target += 0.02 * self.action_noise

        # Clip to joint limits.
        target = np.clip(target, self.jnt_range[:, 0], self.jnt_range[:, 1])

        # Derive a policy-order action vector for logging.
        # action_policy = (target_policy - DEFAULT_JOINT_POS_POLICY) / ACTION_SCALE
        target_policy = target[INV_MAP]
        action_policy = (target_policy - DEFAULT_JOINT_POS_POLICY) / ACTION_SCALE
        return target.astype(np.float32), action_policy.astype(np.float32)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def rotate_vec_by_quat_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vector v into the frame of MuJoCo quaternion q (w,x,y,z)."""
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    return r.inv().apply(v)


# --------------------------------------------------------------------------- #
# Episode runner
# --------------------------------------------------------------------------- #
def run_episode(cfg: EpisodeConfig) -> pd.DataFrame:
    """Run one randomized episode and return a DataFrame of rows."""
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
    controller = HeuristicController(model, cfg, rng)

    # Reset.
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, 0.55]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_adr] = controller.ref
    mujoco.mj_forward(model, data)

    rows: List[dict] = []
    action_policy = np.zeros(N_ACT, dtype=np.float32)

    for t in range(cfg.max_steps):
        target_motor, action_policy = controller.compute(data, pelvis_id)
        data.ctrl[:] = target_motor
        data.xfrc_applied[force_body_id, :3] = cfg.force
        data.xfrc_applied[force_body_id, 3:] = 0.0

        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

        # Logging state.
        base_pos = data.body(pelvis_id).xpos.copy()
        base_quat = data.body(pelvis_id).xquat.copy()
        cvel = data.body(pelvis_id).cvel.copy()
        base_vel_world = cvel[3:]
        base_ang_vel_world = cvel[:3]
        projected_gravity = rotate_vec_by_quat_inv(base_quat, np.array([0.0, 0.0, -1.0]))

        joint_pos_motor = data.qpos[qpos_adr].copy()
        joint_vel_motor = data.qvel[qvel_adr].copy()
        robot_com = data.subtree_com[pelvis_id].copy()
        system_com = robot_com.copy()
        force_app = data.body(force_body_id).xpos.copy()

        row = {
            "episode_id": cfg.episode_id,
            "timestep": t,
            "time": round((t + 1) * STEP_DT, 4),
            "slope_angle_deg": round(cfg.slope_deg, 4),
            "friction": round(cfg.friction, 4),
            "base_pos_x": round(float(base_pos[0]), 6),
            "base_pos_y": round(float(base_pos[1]), 6),
            "base_pos_z": round(float(base_pos[2]), 6),
            "base_quat_w": round(float(base_quat[0]), 6),
            "base_quat_x": round(float(base_quat[1]), 6),
            "base_quat_y": round(float(base_quat[2]), 6),
            "base_quat_z": round(float(base_quat[3]), 6),
            "robot_com_x": round(float(robot_com[0]), 6),
            "robot_com_y": round(float(robot_com[1]), 6),
            "robot_com_z": round(float(robot_com[2]), 6),
            "system_com_x": round(float(system_com[0]), 6),
            "system_com_y": round(float(system_com[1]), 6),
            "system_com_z": round(float(system_com[2]), 6),
            "base_vel_x": round(float(base_vel_world[0]), 6),
            "base_vel_y": round(float(base_vel_world[1]), 6),
            "base_vel_z": round(float(base_vel_world[2]), 6),
            "base_ang_vel_x": round(float(base_ang_vel_world[0]), 6),
            "base_ang_vel_y": round(float(base_ang_vel_world[1]), 6),
            "base_ang_vel_z": round(float(base_ang_vel_world[2]), 6),
            "projected_gravity_x": round(float(projected_gravity[0]), 6),
            "projected_gravity_y": round(float(projected_gravity[1]), 6),
            "projected_gravity_z": round(float(projected_gravity[2]), 6),
            "cmd_vel_x": round(float(cfg.cmd_vel[0]), 4),
            "cmd_vel_y": round(float(cfg.cmd_vel[1]), 4),
            "cmd_yaw_rate": round(float(cfg.cmd_yaw_rate), 4),
            **{f"joint_pos_{i}": round(float(joint_pos_motor[i]), 6) for i in range(N_ACT)},
            **{f"joint_vel_{i}": round(float(joint_vel_motor[i]), 6) for i in range(N_ACT)},
            **{f"joint_default_{i}": round(float(DEFAULT_QPOS_MOTOR[i]), 6) for i in range(N_ACT)},
            **{f"last_action_{i}": round(float(action_policy[i]), 6) for i in range(N_ACT)},
            "force_mag": round(float(np.linalg.norm(cfg.force)), 4),
            "force_x": round(float(cfg.force[0]), 4),
            "force_y": round(float(cfg.force[1]), 4),
            "force_z": round(float(cfg.force[2]), 4),
            "force_body": cfg.force_body,
            "force_app_x": round(float(force_app[0]), 6),
            "force_app_y": round(float(force_app[1]), 6),
            "force_app_z": round(float(force_app[2]), 6),
            "fall_label": 0,
            "steps_to_fall": -1,
        }
        rows.append(row)

        # Fall detection: low base height or extreme orientation.
        rmat = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]).as_matrix()
        pitch = np.arcsin(np.clip(-rmat[2, 0], -1.0, 1.0))
        roll = np.arctan2(rmat[2, 1], rmat[2, 2])
        fallen = (base_pos[2] < 0.42) or (abs(pitch) > 1.0) or (abs(roll) > 1.0)
        if fallen:
            break

    # Label the last 25 steps before a fall.
    if len(rows) < cfg.max_steps:
        fall_idx = len(rows) - 1
        start = max(0, fall_idx - 24)
        for i in range(start, fall_idx + 1):
            rows[i]["fall_label"] = 1
            rows[i]["steps_to_fall"] = fall_idx - i

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Parallel collection
# --------------------------------------------------------------------------- #
def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _run_worker_batch(args: Tuple[int, int, int, str]) -> str:
    start_id, n_eps, base_seed, tmp_path = args
    frames = []
    for i in range(n_eps):
        cfg = EpisodeConfig(episode_id=start_id + i, seed=base_seed + i)
        frames.append(run_episode(cfg))
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(tmp_path, index=False)
    return tmp_path


def run_collector(
    n_episodes: int = 1024,
    n_workers: int = 16,
    output_path: str = "/home/hemad/calhacks/data/g1_mujoco_data.csv",
) -> Tuple[str, int, float]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = n_episodes // n_workers
    rem = n_episodes % n_workers
    tasks: List[Tuple[int, int, int, str]] = []
    cursor = 0
    rng = np.random.default_rng(42)
    for w in range(n_workers):
        count = base + (1 if w < rem else 0)
        tmp = output_path.parent / f"g1_mujoco_data_part_{w:03d}.csv"
        tasks.append((cursor, count, int(rng.integers(0, 1_000_000_000)), str(tmp)))
        cursor += count

    start = time.time()
    from multiprocessing import get_context

    ctx = get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init_worker) as pool:
        tmp_paths = pool.map(_run_worker_batch, tasks)

    dfs = [pd.read_csv(p) for p in tmp_paths]
    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(output_path, index=False)
    for p in tmp_paths:
        Path(p).unlink(missing_ok=True)
    elapsed = time.time() - start
    return str(output_path), len(full), elapsed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--output",
        type=str,
        default="/home/hemad/calhacks/data/g1_mujoco_data.csv",
    )
    args = parser.parse_args()

    path, rows, elapsed = run_collector(args.episodes, args.workers, args.output)
    eps = pd.read_csv(path)["episode_id"].nunique()
    print(f"Saved {path}")
    print(f"Rows: {rows} | Episodes: {eps} | Time: {elapsed:.1f}s")
