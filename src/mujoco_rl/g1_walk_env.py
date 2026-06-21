"""Gymnasium environment for training a G1 walking policy in MuJoCo."""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.spatial.transform import Rotation as R
from src.mujoco_collector.collector import (
    ACTION_SCALE,
    DEFAULT_JOINT_POS_POLICY,
    DEFAULT_QPOS_MOTOR,
    INV_MAP,
    JOINT_NAMES,
    STEP_DT,
    build_model,
)


class G1WalkEnv(gym.Env):
    """MuJoCo G1 velocity-tracking environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        slope_deg: float = 0.0,
        friction: float = 1.0,
        cmd_vel: np.ndarray | None = None,
        max_steps: int = 300,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.slope_deg = slope_deg
        self.friction = friction
        self.cmd_vel = cmd_vel if cmd_vel is not None else np.array([0.5, 0.0], dtype=np.float32)
        self.cmd_yaw_rate = 0.0
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.model = build_model(np.deg2rad(slope_deg), friction)
        self.data = self._new_data()

        self.joint_ids = np.array([self.model.joint(n).id for n in JOINT_NAMES], dtype=np.int32)
        self.qpos_adr = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.qvel_adr = np.array([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.pelvis_id = self.model.body("pelvis").id

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(29,), dtype=np.float32)
        # Obs: projected gravity 3, base ang vel 3, commands 3, joint pos rel 29, joint vel rel 29, last action 29.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(96,), dtype=np.float32)

        self.n_substeps = int(round(STEP_DT / self.model.opt.timestep))
        self.last_action = np.zeros(29, dtype=np.float32)
        self.steps = 0

    def _new_data(self):
        return self._make_data(self.model)

    def _make_data(self, model):
        data = mujoco.MjData(model)
        model.opt.timestep = 0.002
        return data

    def _reset_state(self):
        self.data = self._new_data()
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:3] = [0.0, 0.0, 0.70]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        # Slightly bent knees.
        self.data.qpos[self.qpos_adr] = DEFAULT_QPOS_MOTOR.copy()
        self.data.qpos[self.qpos_adr[[3, 9]]] += 0.2  # knees
        self.data.qpos[self.qpos_adr[[4, 10]]] -= 0.1  # ankles
        self.last_action = np.zeros(29, dtype=np.float32)
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._reset_state()
        return self._get_obs(), {}

    def _get_obs(self):
        base_quat = self.data.body(self.pelvis_id).xquat.copy()
        r = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        projected_gravity = r.inv().apply([0.0, 0.0, -1.0])
        cvel = self.data.body(self.pelvis_id).cvel.copy()
        base_ang_vel = cvel[:3]

        joint_pos_motor = self.data.qpos[self.qpos_adr].copy()
        joint_vel_motor = self.data.qvel[self.qvel_adr].copy()
        joint_pos_policy = joint_pos_motor[INV_MAP]
        joint_vel_policy = joint_vel_motor[INV_MAP]

        obs = np.concatenate([
            base_ang_vel * 0.2,
            projected_gravity,
            [self.cmd_vel[0], self.cmd_vel[1], self.cmd_yaw_rate],
            joint_pos_policy - DEFAULT_JOINT_POS_POLICY,
            joint_vel_policy * 0.05,
            self.last_action,
        ]).astype(np.float32)
        return obs

    def _compute_reward(self, action: np.ndarray):
        base_pos = self.data.body(self.pelvis_id).xpos.copy()
        cvel = self.data.body(self.pelvis_id).cvel.copy()
        base_quat = self.data.body(self.pelvis_id).xquat.copy()
        rmat = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]).as_matrix()
        pitch = float(np.arcsin(np.clip(-rmat[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(rmat[2, 1], rmat[2, 2]))

        alive = 2.0
        track = -2.0 * abs(cvel[3] - self.cmd_vel[0]) - 2.0 * abs(cvel[4] - self.cmd_vel[1]) - 1.0 * abs(cvel[2] - self.cmd_yaw_rate)
        upright = -1.0 * (pitch ** 2 + roll ** 2)
        height = -2.0 * (base_pos[2] - 0.70) ** 2
        energy = -0.0001 * np.sum(self.data.ctrl ** 2)
        action_rate = -0.05 * np.sum((action - self.last_action) ** 2)
        return alive + track + upright + height + energy + action_rate

    def _terminated(self):
        base_pos = self.data.body(self.pelvis_id).xpos.copy()
        base_quat = self.data.body(self.pelvis_id).xquat.copy()
        rmat = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]).as_matrix()
        pitch = float(np.arcsin(np.clip(-rmat[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(rmat[2, 1], rmat[2, 2]))
        return (base_pos[2] < 0.42) or (abs(pitch) > 1.0) or (abs(roll) > 1.0)

    def step(self, action: np.ndarray):
        self.last_action = action.astype(np.float32)
        target_policy = DEFAULT_JOINT_POS_POLICY + action * ACTION_SCALE
        target_motor = target_policy[INV_MAP]
        self.data.ctrl[:] = target_motor

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        reward = self._compute_reward(action)
        terminated = self._terminated()
        truncated = self.steps >= self.max_steps
        self.steps += 1

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, {}

    def render(self):
        return None
