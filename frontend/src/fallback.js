/**
 * Offline fallback — JS mirror of backend/inference.py (G1 23-dof walk). Used
 * ONLY when the FastAPI backend is unreachable, so `npm run dev` alone still
 * demonstrates the factor → motion mapping. The backend is the source of truth;
 * keep this in sync if you edit the gait there.
 */
const STANDING_HEIGHT = 0.7;
const S_HIP_FWD = -1.0;
const S_KNEE = 1.0;
const S_ARM = 1.0;

function profileZ(x, inclineDeg, slopes) {
  let z = Math.tan((inclineDeg * Math.PI) / 180) * x;
  for (const [cx, amp, w] of slopes) z += amp * Math.exp(-(((x - cx) / w) ** 2));
  return z;
}

function leg(phase, aHip, kBase, kAmp, slope) {
  const sw = Math.sin(phase);
  const hip = S_HIP_FWD * aHip * sw;
  const knee = S_KNEE * (kBase + kAmp * Math.max(0, sw));
  let ankle = -0.45 * hip - 0.15 * knee + slope;
  ankle = Math.max(-0.87, Math.min(0.52, ankle));
  return [hip, knee, ankle];
}

function pose(t, gaitF, aHip, kBase, kAmp, aArm, jitter, slope, carry) {
  const pL = 2 * Math.PI * gaitF * t;
  const pR = pL + Math.PI;
  const [lh, lk, la0] = leg(pL, aHip, kBase, kAmp, slope);
  const [rh, rk, ra0] = leg(pR, aHip, kBase, kAmp, slope);
  const la = la0 + jitter * Math.sin(11 * pL);
  const ra = ra0 + jitter * Math.sin(11 * pR);
  const arms = carry
    ? {
        left_shoulder_pitch_joint: -0.38, left_shoulder_roll_joint: 0.06,
        left_shoulder_yaw_joint: 0, left_elbow_joint: 1.6, left_wrist_roll_joint: 0,
        right_shoulder_pitch_joint: -0.38, right_shoulder_roll_joint: -0.06,
        right_shoulder_yaw_joint: 0, right_elbow_joint: 1.6, right_wrist_roll_joint: 0,
      }
    : {
        left_shoulder_pitch_joint: S_ARM * aArm * Math.sin(pL), left_shoulder_roll_joint: 0.18,
        left_shoulder_yaw_joint: 0, left_elbow_joint: 0.35, left_wrist_roll_joint: 0,
        right_shoulder_pitch_joint: S_ARM * aArm * Math.sin(pR), right_shoulder_roll_joint: -0.18,
        right_shoulder_yaw_joint: 0, right_elbow_joint: 0.35, right_wrist_roll_joint: 0,
      };
  const j = {
    left_hip_pitch_joint: lh, left_hip_roll_joint: 0.02, left_hip_yaw_joint: 0,
    left_knee_joint: lk, left_ankle_pitch_joint: la, left_ankle_roll_joint: 0,
    right_hip_pitch_joint: rh, right_hip_roll_joint: -0.02, right_hip_yaw_joint: 0,
    right_knee_joint: rk, right_ankle_pitch_joint: ra, right_ankle_roll_joint: 0,
    waist_yaw_joint: 0.05 * Math.sin(pL),
    ...arms,
  };
  for (const k in j) j[k] = +j[k].toFixed(5);
  return j;
}

export function generateFallbackRollout(p) {
  const { incline_deg, payload_kg, friction, num_slopes, speed_mps, seconds = 8 } = p;
  const fps = 30;

  const slip = Math.max(0.2, Math.min(1, friction / 0.6));
  const length = Math.max(0.5, speed_mps * seconds * slip) + 1.0;

  const slopes = [];
  for (let i = 0; i < num_slopes; i++) {
    const cx = (length * (i + 1)) / (num_slopes + 1);
    slopes.push([cx, 0.1 + 0.04 * (i % 2), Math.max(0.25, length / (num_slopes * 3))]);
  }

  const gaitF = 0.7 + speed_mps * 0.9;
  const ampScale = Math.max(0.4, 1 - 0.02 * payload_kg);
  const aHip = (0.3 + 0.12 * speed_mps) * ampScale;
  const kAmp = (0.55 + 0.18 * speed_mps) * ampScale;
  const kBase = 0.1 + 0.015 * payload_kg;
  const aArm = 0.25 * ampScale;
  const jitter = (1 - slip) * 0.12;
  const lean = 0.3 * (incline_deg * Math.PI) / 180;
  const height = STANDING_HEIGHT - 0.004 * payload_kg - 0.04 * kBase;
  const carry = payload_kg > 0;

  const n = Math.round(seconds * fps);
  const frames = [];
  for (let kf = 0; kf <= n; kf++) {
    const t = kf / fps;
    const x = speed_mps * t * slip;
    const groundZ = profileZ(x, incline_deg, slopes);
    const dz = profileZ(x + 0.05, incline_deg, slopes) - profileZ(x - 0.05, incline_deg, slopes);
    const slope = Math.atan2(dz, 0.1);
    const pL = 2 * Math.PI * gaitF * t;
    const bob = 0.015 * ampScale * Math.cos(2 * pL);
    frames.push({
      t: +t.toFixed(4),
      joints: pose(t, gaitF, aHip, kBase, kAmp, aArm, jitter, slope, carry),
      root: {
        pos: [+x.toFixed(5), 0, +(groundZ + height + bob).toFixed(5)],
        quat: [0, +Math.sin(lean / 2).toFixed(6), 0, +Math.cos(lean / 2).toFixed(6)],
      },
      objects: [],
    });
  }

  const samples = 96;
  const profile = [];
  for (let i = 0; i <= samples; i++) {
    const px = (length * i) / samples;
    profile.push([+px.toFixed(4), +profileZ(px, incline_deg, slopes).toFixed(4)]);
  }

  return {
    params: p,
    terrain: { incline_deg, friction, length: +length.toFixed(3), width: 1.4, profile },
    frames,
    source: 'fallback',
  };
}
