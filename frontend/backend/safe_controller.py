"""Heuristic controller + lightweight MPC safety layer.

At each step the controller proposes a nominal action from the heuristic
balance controller, then evaluates a small set of perturbed actions by
simulating them forward in a copied MuJoCo state and scoring the predicted
future window with the trained fall-risk GRU. The lowest-risk action is applied.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import mujoco
    from scipy.spatial.transform import Rotation as R
    from src.mujoco_collector.collector import (
        ACTION_SCALE,
        DEFAULT_JOINT_POS_POLICY,
        DEFAULT_QPOS_MOTOR,
        INV_MAP,
        JOINT_NAMES,
        STEP_DT,
        HeuristicController,
        rotate_vec_by_quat_inv,
    )
    from risk_model import score_window

    MUJOCO_AVAILABLE = True
except Exception as exc:
    MUJOCO_AVAILABLE = False
    MUJOCO_ERROR = str(exc)


def _state_to_features(
    data: "mujoco.MjData",
    model: "mujoco.MjModel",
    cfg,
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
        cfg.slope_deg,
        cfg.friction,
        float(base_pos[0]),
        float(base_pos[1]),
        float(base_pos[2]),
        float(base_quat[0]),
        float(base_quat[1]),
        float(base_quat[2]),
        float(base_quat[3]),
        float(base_pos[0]),  # robot_com
        float(base_pos[1]),
        float(base_pos[2]),
        float(base_pos[0]),  # system_com
        float(base_pos[1]),
        float(base_pos[2]),
        float(cvel[3]),
        float(cvel[4]),
        float(cvel[5]),
        float(cvel[0]),
        float(cvel[1]),
        float(cvel[2]),
        float(projected_gravity[0]),
        float(projected_gravity[1]),
        float(projected_gravity[2]),
        float(cfg.cmd_vel[0]),
        float(cfg.cmd_vel[1]),
        float(cfg.cmd_yaw_rate),
    ]
    for i in range(len(JOINT_NAMES)):
        values.extend([float(joint_pos_motor[i]), float(joint_vel_motor[i]), float(DEFAULT_QPOS_MOTOR[i]), float(action_policy[i])])
    values.extend([
        float(np.linalg.norm(cfg.force)),
        float(cfg.force[0]),
        float(cfg.force[1]),
        float(cfg.force[2]),
        float(force_app[0]),
        float(force_app[1]),
        float(force_app[2]),
    ])
    return np.array(values, dtype=np.float32)


class SafeController:
    """Heuristic controller with a short-horizon MPC safety wrapper."""

    def __init__(
        self,
        model: "mujoco.MjModel",
        cfg,
        rng: np.random.Generator,
        horizon: int = 3,
        enable_mpc: bool = True,
    ):
        if not MUJOCO_AVAILABLE:
            raise RuntimeError(MUJOCO_ERROR)
        self.model = model
        self.cfg = cfg
        self.rng = rng
        self.enable_mpc = enable_mpc
        self.horizon = horizon

        self.heuristic = HeuristicController(model, cfg, rng)
        self.ref = self.heuristic.ref
        self.joint_ids = np.array([model.joint(n).id for n in JOINT_NAMES], dtype=np.int32)
        self.qpos_adr = np.array([model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.qvel_adr = np.array([model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.pelvis_id = model.body("pelvis").id
        self.torso_id = model.body("torso_link").id
        self.force_body_id = self.pelvis_id if cfg.force_body == "pelvis" else self.torso_id

        self.feature_history: list[np.ndarray] = []
        self.last_action = np.zeros(len(JOINT_NAMES), dtype=np.float32)

    def _candidates(self, nominal: np.ndarray) -> dict[str, np.ndarray]:
        cands = {"nominal": nominal.copy()}
        # Perturbations in motor order: legs are indices 0-11, then waist 12-14, arms 15-28.
        # Lean forward: flex hips, extend ankles slightly.
        forward = nominal.copy()
        forward[0] += 0.05
        forward[6] += 0.05
        forward[4] -= 0.03
        forward[10] -= 0.03
        cands["lean_forward"] = forward

        # Lean back.
        back = nominal.copy()
        back[0] -= 0.05
        back[6] -= 0.05
        back[4] += 0.03
        back[10] += 0.03
        cands["lean_back"] = back

        # Lower COM: bend knees.
        crouch = nominal.copy()
        crouch[3] += 0.08
        crouch[9] += 0.08
        crouch[4] -= 0.05
        crouch[10] -= 0.05
        cands["crouch"] = crouch

        # Widen stance: hip roll out.
        wide = nominal.copy()
        wide[1] += 0.04
        wide[7] -= 0.04
        cands["wide"] = wide

        # Slow down: reduce velocity feedforward effect by damping hip/ankle offsets.
        slow = nominal.copy()
        slow[0] *= 0.8
        slow[6] *= 0.8
        slow[4] *= 0.8
        slow[10] *= 0.8
        cands["slow"] = slow

        return cands

    def _score_action(self, target_motor: np.ndarray, n_substeps: int) -> float:
        """Simulate target_motor for horizon steps and return max predicted risk."""
        copy_data = mujoco.MjData(self.model)
        mujoco.mj_copyData(copy_data, self.model, self._current_data)
        tmp_action = (target_motor[INV_MAP] - DEFAULT_JOINT_POS_POLICY) / ACTION_SCALE
        scores = []
        local_history = list(self.feature_history)
        for _ in range(self.horizon):
            copy_data.ctrl[:] = target_motor
            for _ in range(n_substeps):
                mujoco.mj_step(self.model, copy_data)
            feat = _state_to_features(
                copy_data, self.model, self.cfg, self.pelvis_id, self.force_body_id,
                self.joint_ids, self.qpos_adr, self.qvel_adr, tmp_action,
            )
            local_history.append(feat)
            scores.append(score_window(np.array(local_history[-10:], dtype=np.float32)))
            if len(local_history) > 10:
                local_history = local_history[-10:]
        return float(np.max(scores)) if scores else 1.0

    def compute(self, data: "mujoco.MjData", pelvis_id: int):
        self._current_data = data
        nominal, _ = self.heuristic.compute(data, pelvis_id)

        n_substeps = int(round(STEP_DT / self.model.opt.timestep))
        if not self.enable_mpc or len(self.feature_history) < 3:
            best = nominal
        else:
            candidates = self._candidates(nominal)
            best_name, best_score = None, float("inf")
            for name, target in candidates.items():
                try:
                    s = self._score_action(target, n_substeps)
                except Exception:
                    s = 1.0
                if s < best_score:
                    best_score = s
                    best_name = name
                    best = target
            # print(f"[{len(self.feature_history)}] chose {best_name} score {best_score:.3f}")

        target_policy = best[INV_MAP]
        self.last_action = ((target_policy - DEFAULT_JOINT_POS_POLICY) / ACTION_SCALE).astype(np.float32)

        # Update real history with the state *after* applying best.
        data.ctrl[:] = best
        # Note: the caller is responsible for stepping physics; we record features here before step
        # by using current data (which already reflects prior dynamics). After the caller steps,
        # compute() is called again and history will be appended then.
        feat = _state_to_features(
            data, self.model, self.cfg, self.pelvis_id, self.force_body_id,
            self.joint_ids, self.qpos_adr, self.qvel_adr, self.last_action,
        )
        self.feature_history.append(feat)
        if len(self.feature_history) > 10:
            self.feature_history = self.feature_history[-10:]

        return best.astype(np.float32), self.last_action.copy()
