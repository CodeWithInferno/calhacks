"""Phase-based reference walking controller with balance feedback.

This is a substantial upgrade over the data-collection heuristic: it generates a
sinusoidal biped gait and then adds closed-loop corrections based on pelvis
pitch/roll, COM velocity, and commanded speed. It is still hand-tuned, but it
should be able to walk on flat ground and mild slopes.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.mujoco_collector.collector import (
    ACTION_SCALE,
    DEFAULT_JOINT_POS_POLICY,
    DEFAULT_QPOS_MOTOR,
    INV_MAP,
    JOINT_NAMES,
    STEP_DT,
)

S_HIP_FWD = -1.0
S_KNEE = 1.0
S_ARM = 1.0

# Joint indices in motor order.
L_HIP_PITCH, L_HIP_ROLL, L_HIP_YAW = 0, 1, 2
L_KNEE, L_ANKLE_PITCH, L_ANKLE_ROLL = 3, 4, 5
R_HIP_PITCH, R_HIP_ROLL, R_HIP_YAW = 6, 7, 8
R_KNEE, R_ANKLE_PITCH, R_ANKLE_ROLL = 9, 10, 11
WAIST_YAW, WAIST_ROLL, WAIST_PITCH = 12, 13, 14
L_SHOULDER_PITCH, L_SHOULDER_ROLL, L_SHOULDER_YAW, L_ELBOW = 15, 16, 17, 18
R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW, R_ELBOW = 22, 23, 24, 25


class ReferenceWalkController:
    """Closed-loop reference gait controller."""

    def __init__(self, model, cfg, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.joint_ids = np.array([model.joint(n).id for n in JOINT_NAMES], dtype=np.int32)
        self.jnt_range = np.stack([model.jnt_range[jid] for jid in self.joint_ids])
        self.t = 0.0

        # Gait parameters (tuned for 0.5–1.0 m/s walking).
        self.gait_f = 0.8 + cfg.cmd_vel[0] * 0.5
        self.a_hip = 0.25 + cfg.cmd_vel[0] * 0.12
        self.k_amp = 0.45 + cfg.cmd_vel[0] * 0.18
        self.k_base = 0.15
        self.a_arm = 0.25

        # Reference standing pose used for reset.
        self.ref = DEFAULT_QPOS_MOTOR.copy()
        self.ref[L_KNEE] = 0.25
        self.ref[R_KNEE] = 0.25
        self.ref[L_ANKLE_PITCH] = -0.15
        self.ref[R_ANKLE_PITCH] = -0.15

        # Balance gains.
        self.Kp_pitch = 3.0
        self.Kd_pitch = 0.6
        self.Kp_roll = 3.0
        self.Kd_roll = 0.6
        self.Kp_com_vx = 0.08  # foot placement correction per m/s of COM x velocity
        self.Kp_com_vy = 0.08

    def _leg_targets(self, phase: float, slope: float, com_vx: float, com_vy: float, side: int):
        """Return target (hip_pitch, knee, ankle_pitch) for one leg.

        side: +1 for left, -1 for right (used for lateral foot placement).
        """
        sw = np.sin(phase)
        hip = S_HIP_FWD * self.a_hip * sw
        knee = S_KNEE * (self.k_base + self.k_amp * max(0.0, sw))
        ankle = -0.45 * hip - 0.15 * knee + slope

        # Foot placement capture-point correction during swing.
        if sw > 0.0:
            hip += S_HIP_FWD * self.Kp_com_vx * com_vx
            hip_roll = side * self.Kp_com_vy * abs(com_vy)
        else:
            hip_roll = 0.0

        # Swing foot clearance.
        if sw > 0.0:
            ankle += 0.12 * sw

        return hip, knee, ankle, hip_roll

    def compute(self, data, pelvis_id: int):
        # Advance time.
        self.t += STEP_DT
        phase = 2.0 * np.pi * self.gait_f * self.t

        # State feedback.
        base_quat = data.body(pelvis_id).xquat
        rmat = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]).as_matrix()
        pitch = float(np.arcsin(np.clip(-rmat[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(rmat[2, 1], rmat[2, 2]))
        cvel = data.body(pelvis_id).cvel
        com_vx = float(cvel[3])
        com_vy = float(cvel[4])

        # Desired lean into the hill.
        slope_rad = np.deg2rad(self.cfg.slope_deg)
        pitch_des = 0.15 * slope_rad
        pitch_err = pitch - pitch_des
        roll_err = roll

        tau_pitch = self.Kp_pitch * pitch_err + self.Kd_pitch * float(cvel[0])
        tau_roll = self.Kp_roll * roll_err + self.Kd_roll * float(cvel[1])

        target = np.zeros(29, dtype=np.float32)

        # Left leg (phase).
        l_hp, l_k, l_ap, l_hr = self._leg_targets(phase, slope_rad, com_vx, com_vy, side=1)
        target[L_HIP_PITCH] = l_hp + tau_pitch
        target[L_KNEE] = l_k
        target[L_ANKLE_PITCH] = l_ap - 1.2 * tau_pitch
        target[L_HIP_ROLL] = 0.04 - tau_roll + l_hr
        target[L_HIP_YAW] = 0.0
        target[L_ANKLE_ROLL] = 0.5 * tau_roll

        # Right leg (phase + pi).
        r_hp, r_k, r_ap, r_hr = self._leg_targets(phase + np.pi, slope_rad, com_vx, com_vy, side=-1)
        target[R_HIP_PITCH] = r_hp + tau_pitch
        target[R_KNEE] = r_k
        target[R_ANKLE_PITCH] = r_ap - 1.2 * tau_pitch
        target[R_HIP_ROLL] = -0.04 + tau_roll + r_hr
        target[R_HIP_YAW] = 0.0
        target[R_ANKLE_ROLL] = -0.5 * tau_roll

        # Waist counter-rotation.
        target[WAIST_YAW] = 0.05 * np.sin(phase)
        target[WAIST_ROLL] = 0.0
        target[WAIST_PITCH] = -0.3 * slope_rad

        # Arms swing opposite to legs.
        pL = phase
        pR = phase + np.pi
        target[L_SHOULDER_PITCH] = S_ARM * self.a_arm * np.sin(pL) - 0.2 * slope_rad
        target[L_SHOULDER_ROLL] = 0.18
        target[L_SHOULDER_YAW] = 0.0
        target[L_ELBOW] = 0.35
        target[R_SHOULDER_PITCH] = S_ARM * self.a_arm * np.sin(pR) - 0.2 * slope_rad
        target[R_SHOULDER_ROLL] = -0.18
        target[R_SHOULDER_YAW] = 0.0
        target[R_ELBOW] = 0.35

        # Wrists stay at default.
        for idx in [19, 20, 21, 26, 27, 28]:
            target[idx] = DEFAULT_QPOS_MOTOR[idx]

        # Joint limits.
        target = np.clip(target, self.jnt_range[:, 0], self.jnt_range[:, 1])

        # Policy-order action vector for logging.
        target_policy = target[INV_MAP]
        action_policy = (target_policy - DEFAULT_JOINT_POS_POLICY) / ACTION_SCALE
        return target.astype(np.float32), action_policy.astype(np.float32)
