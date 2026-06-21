"""
inference.py — turn the 5 UI factors into a Unitree G1 (23-dof) walking rollout.

  incline_deg  ground incline along the travel direction (+X)
  payload_kg   mass the robot carries (heavier -> shorter, more crouched stride)
  friction     ground friction (low -> slip: less forward progress + gait jitter)
  num_slopes   number of slope/bump features along the path
  speed_mps    commanded locomotion speed (sets stride frequency + length)

The gait here is a PROCEDURAL placeholder (sinusoidal biped walk) so the viewer
shows the real G1 moving before you have trained rollouts. It is NOT foot-IK'd,
so on steep/bumpy terrain feet may float or clip — real Isaac Lab rollouts will
have correct contact.

================================  INTEGRATION SEAM  ============================
`run_rollout()` is where your TRAINED ISAAC LAB POLICY plugs in. Replace the call
to `generate_stub_rollout()` with real inference (build the env with these domain
params, load the checkpoint from Nebius, step the policy, record per-frame joint
positions + base pose) and return the SAME dict schema. Nothing downstream changes.
===============================================================================
"""

import math

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
    """Return (hip_pitch, knee, ankle_pitch) for one leg at gait `phase`.

    `slope` (rad) is the local terrain incline; it's added to the ankle so the
    sole stays flat on the ground on inclines/slopes (like a human ankle
    dorsiflexing uphill). Ankle is clamped to the URDF limits."""
    sw = math.sin(phase)
    hip = S_HIP_FWD * a_hip * sw
    knee = S_KNEE * (k_base + k_amp * max(0.0, sw))     # flex during forward swing
    ankle = -0.45 * hip - 0.15 * knee + slope           # level foot + match slope
    ankle = max(-0.87, min(0.52, ankle))
    return hip, knee, ankle


def _pose(t, gait_f, a_hip, k_base, k_amp, a_arm, jitter, slope=0.0, carry=False):
    """Full 23-joint dict at time t."""
    pL = 2 * math.pi * gait_f * t
    pR = pL + math.pi

    lh, lk, la = _leg(pL, a_hip, k_base, k_amp, slope)
    rh, rk, ra = _leg(pR, a_hip, k_base, k_amp, slope)

    # Low-friction slipping adds a little high-frequency noise to the ankles.
    la += jitter * math.sin(11.0 * pL)
    ra += jitter * math.sin(11.0 * pR)

    if carry:
        # Forearms forward and together so the hands cradle the 1 ft carton in
        # front of the waist; elbows ~bent, upper arms close to the body.
        arms = {
            "left_shoulder_pitch_joint": -0.38, "left_shoulder_roll_joint": 0.06,
            "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 1.6, "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": -0.38, "right_shoulder_roll_joint": -0.06,
            "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 1.6, "right_wrist_roll_joint": 0.0,
        }
    else:
        # Arms swing opposite the same-side leg.
        arms = {
            "left_shoulder_pitch_joint": S_ARM * a_arm * math.sin(pL), "left_shoulder_roll_joint": 0.18,
            "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.35, "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": S_ARM * a_arm * math.sin(pR), "right_shoulder_roll_joint": -0.18,
            "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.35, "right_wrist_roll_joint": 0.0,
        }

    j = {
        # left leg
        "left_hip_pitch_joint": lh,
        "left_hip_roll_joint": 0.02,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": lk,
        "left_ankle_pitch_joint": la,
        "left_ankle_roll_joint": 0.0,
        # right leg
        "right_hip_pitch_joint": rh,
        "right_hip_roll_joint": -0.02,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": rk,
        "right_ankle_pitch_joint": ra,
        "right_ankle_roll_joint": 0.0,
        # waist counter-rotation
        "waist_yaw_joint": 0.05 * math.sin(pL),
        **arms,
    }
    return {k: round(v, 5) for k, v in j.items()}


def generate_stub_rollout(p):
    incline = p["incline_deg"]
    payload = p["payload_kg"]
    friction = p["friction"]
    n_slopes = int(p["num_slopes"])
    speed = p["speed_mps"]
    seconds = p["seconds"]
    fps = 30

    slip = max(0.2, min(1.0, friction / 0.6))
    length = max(0.5, speed * seconds * slip) + 1.0

    slopes = []
    for i in range(n_slopes):
        cx = length * (i + 1) / (n_slopes + 1)
        amp = 0.10 + 0.04 * (i % 2)
        w = max(0.25, length / (n_slopes * 3))
        slopes.append((cx, amp, w))

    # Gait shaping from the factors.
    gait_f = 0.7 + speed * 0.9                       # stride frequency
    amp_scale = max(0.4, 1.0 - 0.02 * payload)       # heavier -> shorter stride
    a_hip = (0.30 + 0.12 * speed) * amp_scale
    k_amp = (0.55 + 0.18 * speed) * amp_scale
    k_base = 0.10 + 0.015 * payload                  # heavier -> more crouch
    a_arm = 0.25 * amp_scale
    jitter = (1.0 - slip) * 0.12
    lean = 0.30 * math.radians(incline)              # lean into the hill
    height = STANDING_HEIGHT - 0.004 * payload - 0.04 * k_base
    carry = payload > 0                              # carrying the carton?

    n = int(seconds * fps)
    frames = []
    for kf in range(n + 1):
        t = kf / fps
        x = speed * t * slip
        ground_z = _profile_z(x, incline, slopes)

        # Local terrain slope (rad) under the robot -> feet match the ground.
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

    samples = 96
    profile = [
        [round(length * i / samples, 4),
         round(_profile_z(length * i / samples, incline, slopes), 4)]
        for i in range(samples + 1)
    ]

    return {
        "params": p,
        "terrain": {
            "incline_deg": incline,
            "friction": friction,
            "length": round(length, 3),
            "width": 1.4,
            "profile": profile,
        },
        "frames": frames,
        "source": "stub",
    }


def run_rollout(params: dict) -> dict:
    """Public entry point. Swap the body for real Isaac Lab inference (see top)."""
    return generate_stub_rollout(params)
