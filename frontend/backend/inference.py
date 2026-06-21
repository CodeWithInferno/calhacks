"""
inference.py — generate a rollout and score it with the trained fall-risk GRU.

The backend tries to generate a real MuJoCo rollout driven by our heuristic
controller. If MuJoCo is not installed or the collector cannot be imported, it
falls back to the procedural stub walk (used during early frontend development).

The rollout JSON returned to the browser is unchanged except for an added
`risk_score` field on each frame (0 = safe, 1 = fall imminent).
"""
from __future__ import annotations

import math
from pathlib import Path

import risk_model
import mujoco_rollout

# G1 standing pelvis height (m) with a slight knee bend.
STANDING_HEIGHT = 0.70

# Sign conventions for G1 23dof (axes: hip/knee/ankle/shoulder/elbow pitch about
# +Y; knee & elbow flex POSITIVE per URDF limits). Tuned so the walk reads right.
S_HIP_FWD = -1.0   # negative hip_pitch swings the leg forward
S_KNEE = 1.0       # positive knee flexes
S_ARM = 1.0        # arm swings opposite the same-side leg


def _profile_z(x, incline_deg, slopes):
    z = math.tan(math.radians(incline_deg)) * x
    for cx, amp, w in slopes:
        z += amp * math.exp(-(((x - cx) / w) ** 2))
    return z


def _leg(phase, a_hip, k_base, k_amp, slope):
    sw = math.sin(phase)
    hip = S_HIP_FWD * a_hip * sw
    knee = S_KNEE * (k_base + k_amp * max(0.0, sw))
    ankle = -0.45 * hip - 0.15 * knee + slope
    ankle = max(-0.87, min(0.52, ankle))
    return hip, knee, ankle


def _pose(t, gait_f, a_hip, k_base, k_amp, a_arm, jitter, slope=0.0, carry=False):
    pL = 2 * math.pi * gait_f * t
    pR = pL + math.pi

    lh, lk, la = _leg(pL, a_hip, k_base, k_amp, slope)
    rh, rk, ra = _leg(pR, a_hip, k_base, k_amp, slope)

    la += jitter * math.sin(11.0 * pL)
    ra += jitter * math.sin(11.0 * pR)

    if carry:
        arms = {
            "left_shoulder_pitch_joint": -0.38, "left_shoulder_roll_joint": 0.06,
            "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 1.6, "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": -0.38, "right_shoulder_roll_joint": -0.06,
            "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 1.6, "right_wrist_roll_joint": 0.0,
        }
    else:
        arms = {
            "left_shoulder_pitch_joint": S_ARM * a_arm * math.sin(pL), "left_shoulder_roll_joint": 0.18,
            "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.35, "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": S_ARM * a_arm * math.sin(pR), "right_shoulder_roll_joint": -0.18,
            "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.35, "right_wrist_roll_joint": 0.0,
        }

    j = {
        "left_hip_pitch_joint": lh,
        "left_hip_roll_joint": 0.02,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": lk,
        "left_ankle_pitch_joint": la,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": rh,
        "right_hip_roll_joint": -0.02,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": rk,
        "right_ankle_pitch_joint": ra,
        "right_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.05 * math.sin(pL),
        "waist_roll_joint": 0.0,
        **arms,
    }
    return {k: round(v, 5) for k, v in j.items()}


def _build_terrain_profile(p):
    incline = p["incline_deg"]
    n_slopes = int(p["num_slopes"])
    speed = p["speed_mps"]
    seconds = p["seconds"]
    slip = max(0.2, min(1.0, p["friction"] / 0.6))
    length = max(0.5, speed * seconds * slip) + 1.0

    slopes = []
    for i in range(n_slopes):
        cx = length * (i + 1) / (n_slopes + 1)
        amp = 0.10 + 0.04 * (i % 2)
        w = max(0.25, length / (n_slopes * 3))
        slopes.append((cx, amp, w))

    samples = 96
    profile = [
        [round(length * i / samples, 4), round(_profile_z(length * i / samples, incline, slopes), 4)]
        for i in range(samples + 1)
    ]
    return {
        "incline_deg": incline,
        "friction": p["friction"],
        "length": round(length, 3),
        "width": 1.4,
        "profile": profile,
    }, slopes, length


def generate_stub_rollout(p):
    incline = p["incline_deg"]
    payload = p["payload_kg"]
    friction = p["friction"]
    n_slopes = int(p["num_slopes"])
    speed = p["speed_mps"]
    seconds = p["seconds"]

    terrain, slopes, length = _build_terrain_profile(p)

    gait_f = 0.7 + speed * 0.9
    amp_scale = max(0.4, 1.0 - 0.02 * payload)
    a_hip = (0.30 + 0.12 * speed) * amp_scale
    k_amp = (0.55 + 0.18 * speed) * amp_scale
    k_base = 0.10 + 0.015 * payload
    a_arm = 0.25 * amp_scale
    jitter = (1.0 - max(0.2, min(1.0, friction / 0.6))) * 0.12
    lean = 0.30 * math.radians(incline)
    height = STANDING_HEIGHT - 0.004 * payload - 0.04 * k_base
    carry = payload > 0

    fps = 30
    n = int(seconds * fps)
    frames = []
    for kf in range(n + 1):
        t = kf / fps
        x = speed * t * max(0.2, min(1.0, friction / 0.6))
        ground_z = _profile_z(x, incline, slopes)
        dz = _profile_z(x + 0.05, incline, slopes) - _profile_z(x - 0.05, incline, slopes)
        slope = math.atan2(dz, 0.10)

        pL = 2 * math.pi * gait_f * t
        bob = 0.015 * amp_scale * math.cos(2 * pL)
        z = ground_z + height + bob
        qy, qw = math.sin(lean / 2), math.cos(lean / 2)

        frames.append({
            "t": round(t, 4),
            "joints": _pose(t, gait_f, a_hip, k_base, k_amp, a_arm, jitter, slope, carry),
            "root": {
                "pos": [round(x, 5), 0.0, round(z, 5)],
                "quat": [0.0, round(qy, 6), 0.0, round(qw, 6)],
            },
            "objects": [],
        })

    return {"params": p, "terrain": terrain, "frames": frames, "source": "stub"}


def run_rollout(params: dict) -> dict:
    """Public entry point. Uses real MuJoCo when possible, stub otherwise."""
    seed = int(params.get("seed", 42))

    if mujoco_rollout.MUJOCO_AVAILABLE:
        try:
            frames, df, _ = mujoco_rollout.generate_rollout(params, controller_type="safe")
            terrain, _, _ = _build_terrain_profile(params)
            result = {"params": params, "terrain": terrain, "frames": frames, "source": "mujoco_safe"}
        except Exception as exc:
            # Fall back to stub on any MuJoCo error so the frontend never breaks.
            result = generate_stub_rollout(params)
            df = None
            result["mujoco_error"] = str(exc)
    else:
        result = generate_stub_rollout(params)
        df = None

    # Score risk if we have a MuJoCo DataFrame; otherwise no risk field.
    if df is not None and len(df) > 0:
        scores = risk_model.score_dataframe(df)
        # Align scores to frames by time. The first _WINDOW frames have no score.
        frame_times = {i: f["t"] for i, f in enumerate(result["frames"])}
        if len(scores) > 0:
            valid = scores.dropna()
            min_t = df["time"].iloc[0]
            max_t = df["time"].iloc[-1]
            for i, f in enumerate(result["frames"]):
                t = f["t"]
                if t < min_t or t > max_t or len(valid) == 0:
                    f["risk_score"] = None
                else:
                    # nearest score by time
                    idx = int(round((t - min_t) / (max_t - min_t) * (len(valid) - 1)))
                    idx = max(0, min(len(valid) - 1, idx))
                    f["risk_score"] = round(float(valid.iloc[idx]), 4)
        else:
            for f in result["frames"]:
                f["risk_score"] = None

    return result
