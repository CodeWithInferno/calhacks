"""PPO policy controller for the trained G1 walking policy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scipy.spatial.transform import Rotation as R
    from stable_baselines3 import PPO
    from src.mujoco_collector.collector import (
        ACTION_SCALE,
        DEFAULT_JOINT_POS_POLICY,
        INV_MAP,
        JOINT_NAMES,
    )

    SB3_AVAILABLE = True
except Exception as exc:
    SB3_AVAILABLE = False
    SB3_ERROR = str(exc)

DEFAULT_POLICY_PATH = Path("/home/hemad/calhacks/models/g1_ppo_walk_v2/g1_ppo_final.zip")


def _resolve_policy_path(path: Path) -> Path:
    if path.exists():
        return path
    # Fall back to newest checkpoint if final has not been saved yet.
    checkpoints = sorted(path.parent.glob("g1_ppo_*_steps.zip"), key=lambda p: p.stat().st_mtime)
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"PPO policy not found at {path}")


class PPOController:
    """Loads a stable-baselines3 PPO policy and runs inference."""

    def __init__(self, model, cfg, rng, policy_path: Path | str | None = None):
        if not SB3_AVAILABLE:
            raise RuntimeError(f"stable-baselines3 not available: {SB3_ERROR}")
        self.cfg = cfg
        self.rng = rng
        self.policy_path = _resolve_policy_path(Path(policy_path or DEFAULT_POLICY_PATH))

        self.policy = PPO.load(self.policy_path, device="auto")
        self.joint_ids = np.array([model.joint(n).id for n in JOINT_NAMES], dtype=np.int32)
        self.qpos_adr = np.array([model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.qvel_adr = np.array([model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.pelvis_id = model.body("pelvis").id
        self.last_action = np.zeros(29, dtype=np.float32)

    def _get_obs(self, data):
        base_quat = data.body(self.pelvis_id).xquat.copy()
        r = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        projected_gravity = r.inv().apply([0.0, 0.0, -1.0])
        cvel = data.body(self.pelvis_id).cvel.copy()
        base_ang_vel = cvel[:3]

        joint_pos_motor = data.qpos[self.qpos_adr].copy()
        joint_vel_motor = data.qvel[self.qvel_adr].copy()
        joint_pos_policy = joint_pos_motor[INV_MAP]
        joint_vel_policy = joint_vel_motor[INV_MAP]

        obs = np.concatenate([
            base_ang_vel * 0.2,
            projected_gravity,
            [self.cfg.cmd_vel[0], self.cfg.cmd_vel[1], self.cfg.cmd_yaw_rate],
            joint_pos_policy - DEFAULT_JOINT_POS_POLICY,
            joint_vel_policy * 0.05,
            self.last_action,
        ]).astype(np.float32)
        return obs

    def compute(self, data, pelvis_id: int):
        obs = self._get_obs(data)
        action, _ = self.policy.predict(obs, deterministic=True)
        self.last_action = action.astype(np.float32)
        target_policy = DEFAULT_JOINT_POS_POLICY + self.last_action * ACTION_SCALE
        target_motor = target_policy[INV_MAP]
        return target_motor.astype(np.float32), self.last_action.copy()
