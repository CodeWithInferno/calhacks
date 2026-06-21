"""Random search for a parametric walking gait on G1 in MuJoCo."""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
from src.mujoco_collector.collector import (
    ACTION_SCALE,
    DEFAULT_JOINT_POS_POLICY,
    DEFAULT_QPOS_MOTOR,
    INV_MAP,
    JOINT_NAMES,
    STEP_DT,
    EpisodeConfig,
    build_model,
)

# Motor-order indices.
L_HIP_PITCH, L_HIP_ROLL, L_HIP_YAW = 0, 1, 2
L_KNEE, L_ANKLE_PITCH, L_ANKLE_ROLL = 3, 4, 5
R_HIP_PITCH, R_HIP_ROLL, R_HIP_YAW = 6, 7, 8
R_KNEE, R_ANKLE_PITCH, R_ANKLE_ROLL = 9, 10, 11
WAIST_YAW, WAIST_ROLL, WAIST_PITCH = 12, 13, 14
L_SHOULDER_PITCH, L_SHOULDER_ROLL, L_SHOULDER_YAW, L_ELBOW = 15, 16, 17, 18
R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW, R_ELBOW = 22, 23, 24, 25


@dataclass(frozen=True)
class GaitParams:
    freq: float = 1.0
    hip_amp: float = 0.25
    hip_sign: float = -1.0
    knee_amp: float = 0.45
    knee_phase: float = 0.0
    knee_base: float = 0.15
    ankle_gain: float = -0.45
    ankle_offset: float = -0.15
    hip_roll_amp: float = 0.04
    vx_feedforward: float = 0.1
    Kp_pitch: float = 2.0
    Kd_pitch: float = 0.5
    Kp_roll: float = 2.0
    Kd_roll: float = 0.5
    Kp_vx: float = 0.1
    Kp_vy: float = 0.1
    shoulder_amp: float = 0.25
    waist_yaw_amp: float = 0.05


class TunedGaitController:
    def __init__(self, model, cfg: EpisodeConfig, params: GaitParams):
        self.cfg = cfg
        self.p = params
        self.t = 0.0
        self.joint_ids = np.array([model.joint(n).id for n in JOINT_NAMES], dtype=np.int32)
        self.qpos_adr = np.array([model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.qvel_adr = np.array([model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.jnt_range = np.stack([model.jnt_range[jid] for jid in self.joint_ids])
        self.ref = DEFAULT_QPOS_MOTOR.copy()
        self.ref[[L_KNEE, R_KNEE]] += 0.25
        self.ref[[L_ANKLE_PITCH, R_ANKLE_PITCH]] -= 0.15

    def compute(self, data, pelvis_id: int):
        self.t += STEP_DT
        phase = 2.0 * np.pi * self.p.freq * self.t

        base_quat = data.body(pelvis_id).xquat
        rmat = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]).as_matrix()
        pitch = float(np.arcsin(np.clip(-rmat[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(rmat[2, 1], rmat[2, 2]))
        cvel = data.body(pelvis_id).cvel
        # World-frame velocities from freejoint.
        vx = float(data.qvel[0])
        vy = float(data.qvel[1])

        slope_rad = np.deg2rad(self.cfg.slope_deg)
        pitch_des = 0.15 * slope_rad
        roll_des = 0.0

        tau_pitch = self.p.Kp_pitch * (pitch - pitch_des) + self.p.Kd_pitch * float(cvel[0])
        tau_roll = self.p.Kp_roll * (roll - roll_des) + self.p.Kd_roll * float(cvel[1])

        vx_err = vx - self.cfg.cmd_vel[0]
        vy_err = vy - self.cfg.cmd_vel[1]
        foot_x = -self.p.Kp_vx * vx_err
        foot_y = -self.p.Kp_vy * vy_err

        target = np.zeros(29, dtype=np.float32)

        def _leg(ph: float, side: int):
            s = np.sin(ph)
            # Forward feedforward + velocity feedback (sign-corrected for hip direction).
            hip_ff = self.p.hip_sign * self.cfg.cmd_vel[0] * self.p.vx_feedforward
            hip_vcorr = self.p.hip_sign * (-self.p.Kp_vx * vx_err)
            hip = self.p.hip_sign * self.p.hip_amp * s + tau_pitch + hip_ff + hip_vcorr
            knee = self.p.knee_base + max(0.0, self.p.knee_amp * np.sin(ph + self.p.knee_phase))
            ankle = self.p.ankle_gain * hip + self.p.ankle_offset + tau_pitch
            hip_roll = side * self.p.hip_roll_amp - side * tau_roll - side * self.p.Kp_vy * vy_err
            return hip, knee, ankle, hip_roll

        l_hp, l_k, l_ap, l_hr = _leg(phase, 1)
        target[L_HIP_PITCH] = l_hp
        target[L_KNEE] = l_k
        target[L_ANKLE_PITCH] = l_ap
        target[L_HIP_ROLL] = l_hr
        target[L_HIP_YAW] = 0.0
        target[L_ANKLE_ROLL] = 0.5 * tau_roll

        r_hp, r_k, r_ap, r_hr = _leg(phase + np.pi, -1)
        target[R_HIP_PITCH] = r_hp
        target[R_KNEE] = r_k
        target[R_ANKLE_PITCH] = r_ap
        target[R_HIP_ROLL] = r_hr
        target[R_HIP_YAW] = 0.0
        target[R_ANKLE_ROLL] = -0.5 * tau_roll

        target[WAIST_YAW] = self.p.waist_yaw_amp * np.sin(phase)
        target[WAIST_ROLL] = 0.0
        target[WAIST_PITCH] = -0.3 * slope_rad

        pL, pR = phase, phase + np.pi
        target[L_SHOULDER_PITCH] = self.p.shoulder_amp * np.sin(pL) - 0.2 * slope_rad
        target[L_SHOULDER_ROLL] = 0.18
        target[L_SHOULDER_YAW] = 0.0
        target[L_ELBOW] = 0.35
        target[R_SHOULDER_PITCH] = self.p.shoulder_amp * np.sin(pR) - 0.2 * slope_rad
        target[R_SHOULDER_ROLL] = -0.18
        target[R_SHOULDER_YAW] = 0.0
        target[R_ELBOW] = 0.35

        for idx in [19, 20, 21, 26, 27, 28]:
            target[idx] = DEFAULT_QPOS_MOTOR[idx]

        target = np.clip(target, self.jnt_range[:, 0] + 0.01, self.jnt_range[:, 1] - 0.01)
        target_policy = target[INV_MAP]
        action_policy = (target_policy - DEFAULT_JOINT_POS_POLICY) / ACTION_SCALE
        return target.astype(np.float32), action_policy.astype(np.float32)


def _is_fallen(df) -> bool:
    if len(df) == 0:
        return True
    return len(df) < 300


def evaluate_params(args: tuple[int, GaitParams]) -> dict[str, Any]:
    idx, params = args
    cfg = EpisodeConfig(
        episode_id=idx,
        seed=1000 + idx,
        max_steps=500,
        slope_deg=0.0,
        friction=1.0,
        cmd_vel=np.array([0.5, 0.0], dtype=np.float32),
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
    joint_ids = np.array([model.joint(n).id for n in JOINT_NAMES], dtype=np.int32)
    qpos_adr = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32)

    controller = TunedGaitController(model, cfg, params)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, 0.70]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_adr] = controller.ref
    mujoco.mj_forward(model, data)

    last_x = 0.0
    last_z = 0.70
    steps = 0
    for t in range(cfg.max_steps):
        target_motor, _ = controller.compute(data, pelvis_id)
        data.ctrl[:] = target_motor
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

        base_pos = data.body(pelvis_id).xpos.copy()
        base_quat = data.body(pelvis_id).xquat.copy()
        rmat = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]).as_matrix()
        pitch = float(np.arcsin(np.clip(-rmat[2, 0], -1.0, 1.0)))
        roll = float(np.arctan2(rmat[2, 1], rmat[2, 2]))
        last_x = float(base_pos[0])
        last_z = float(base_pos[2])
        steps = t + 1
        if (last_z < 0.42) or (abs(pitch) > 1.0) or (abs(roll) > 1.0):
            break

    duration = steps * STEP_DT
    # Prefer controllers that stay alive; among survivors, reward forward distance.
    if duration < 6.0:
        objective = -5.0 + last_x - 2.0 * abs(last_z - 0.70)
    else:
        objective = 10.0 * last_x + 0.5 * duration - 2.0 * abs(last_z - 0.70)
    return {
        "idx": idx,
        "params": asdict(params),
        "distance": last_x,
        "duration": duration,
        "final_height": last_z,
        "objective": objective,
    }


def sample_params(rng: np.random.Generator) -> GaitParams:
    return GaitParams(
        freq=float(rng.uniform(0.6, 1.8)),
        hip_amp=float(rng.uniform(0.2, 0.9)),
        hip_sign=-1.0,
        knee_amp=float(rng.uniform(0.2, 1.2)),
        knee_phase=float(rng.uniform(0.0, np.pi)),
        knee_base=float(rng.uniform(0.05, 0.4)),
        ankle_gain=float(rng.uniform(-0.7, -0.15)),
        ankle_offset=float(rng.uniform(-0.3, 0.05)),
        hip_roll_amp=float(rng.uniform(0.0, 0.12)),
        Kp_pitch=float(rng.uniform(0.5, 4.0)),
        Kd_pitch=float(rng.uniform(0.1, 1.0)),
        Kp_roll=float(rng.uniform(0.5, 4.0)),
        Kd_roll=float(rng.uniform(0.1, 1.0)),
        vx_feedforward=float(rng.uniform(0.0, 0.5)),
        Kp_vx=float(rng.uniform(0.0, 0.6)),
        Kp_vy=float(rng.uniform(0.0, 0.6)),
        shoulder_amp=float(rng.uniform(0.0, 0.4)),
        waist_yaw_amp=float(rng.uniform(0.0, 0.1)),
    )


def main(n_trials: int = 400, n_workers: int = 8):
    rng = np.random.default_rng(42)
    candidates = [sample_params(rng) for _ in range(n_trials)]

    start = time.time()
    with mp.Pool(n_workers) as pool:
        results = pool.map(evaluate_params, enumerate(candidates))

    results.sort(key=lambda r: r["objective"], reverse=True)
    top = results[0]
    print(f"Search done in {time.time() - start:.1f}s")
    print(f"Top objective={top['objective']:.2f} distance={top['distance']:.2f}m duration={top['duration']:.2f}s height={top['final_height']:.2f}m")
    print("Top params:", top["params"])

    out_dir = Path("results/gait_search")
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    (out_dir / "top_params.json").write_text(json.dumps(top, indent=2))
    (out_dir / "all_results.json").write_text(json.dumps(results[:50], indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    main(args.trials, args.workers)
