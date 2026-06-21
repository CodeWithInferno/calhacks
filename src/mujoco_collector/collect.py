"""
MuJoCo-based data collector for G1 fall-risk world model.

Loads Unitree G1 29-DoF in MuJoCo, runs the pre-trained ONNX velocity policy,
randomizes slopes / friction / external forces per episode, and logs per-timestep
state to CSV matching DATA_SCHEMA.md.
"""

import os
import re
import yaml
import argparse
import numpy as np
import pandas as pd
import mujoco
import onnxruntime as ort
from multiprocessing import Pool


# Paths
UNITREE_MUJOCO = "/home/hemad/unitree_mujoco"
G1_SCENE_DIR = os.path.join(UNITREE_MUJOCO, "unitree_robots/g1")
G1_XML = os.path.join(G1_SCENE_DIR, "scene_29dof.xml")
POLICY_ONNX = "/home/hemad/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx"
DEPLOY_YAML = "/home/hemad/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml"

HISTORY_LEN = 5
STEP_DT = 0.02
MAX_STEPS = 500
PREDICTION_HORIZON = 25


def load_deploy_config():
    with open(DEPLOY_YAML) as f:
        return yaml.safe_load(f)


def make_sloped_scene(base_xml, slope_deg, friction, out_path):
    """Generate a scene XML with a sloped box floor."""
    slope_rad = np.deg2rad(slope_deg)
    half_quat = slope_rad / 2.0
    quat = f"{np.cos(half_quat):.6f} 0 {-np.sin(half_quat):.6f} 0"  # rotate around y

    scene = f"""<mujoco model="g1_29dof sloped scene">
  <include file="g1_29dof.xml"/>

  <statistic center="0 0 0.5" extent="2.0"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="box" size="10 10 0.05"
          pos="0 0 -0.05"
          quat="{quat}"
          friction="{friction} 0.005 0.0001"
          material="groundplane"/>
  </worldbody>
</mujoco>
"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(scene)


def add_actuators(model):
    """Add position actuators for each actuated joint in MuJoCo order."""
    # Build actuator string for joints excluding floating_base.
    cfg = load_deploy_config()
    stiffness = cfg["stiffness"]
    damping = cfg["damping"]

    act_lines = []
    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        # actuator index will be in order of appearance
        act_lines.append(
            f'<position name="act_{jnt_name}" joint="{jnt_name}" kp="{stiffness[len(act_lines) % len(stiffness)]}" '
            f'kv="{damping[len(act_lines) % len(damping)]}"/>'
        )

    # We need to edit the model XML to include actuators before loading.
    # Simpler: save a temp XML with actuators appended.
    return act_lines


def load_model_with_actuators(scene_xml):
    """Load MuJoCo model and add position actuators."""
    with open(scene_xml) as f:
        xml_str = f.read()

    # Find </mujoco> and insert actuator block before it.
    act_lines = add_actuators(mujoco.MjModel.from_xml_path(scene_xml))
    actuator_block = "\n  <actuator>\n    " + "\n    ".join(act_lines) + "\n  </actuator>\n"

    xml_str = xml_str.replace("</mujoco>", actuator_block + "</mujoco>")

    tmp_path = os.path.join(G1_SCENE_DIR, "scene_slope_with_actuators.xml")
    with open(tmp_path, "w") as f:
        f.write(xml_str)

    model = mujoco.MjModel.from_xml_path(tmp_path)
    data = mujoco.MjData(model)
    return model, data


def get_actuated_joint_info(model):
    """Return list of actuated joints (name, qposadr, qveladr, ctrlid) in MuJoCo order."""
    joints = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        qposadr = model.jnt_qposadr[i]
        qveladr = model.jnt_dofadr[i]
        joints.append({"name": name, "qposadr": qposadr, "qveladr": qveladr})

    # ctrlid should match the order of actuators we added.
    for idx, j in enumerate(joints):
        j["ctrlid"] = idx

    return joints


def compute_projected_gravity(data):
    """Gravity vector (0,0,-1) rotated into base frame."""
    base_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    rot_mat = data.xmat[base_id].reshape(3, 3)
    gravity_world = np.array([0, 0, -1])
    return rot_mat.T @ gravity_world


def compute_base_ang_vel(data):
    base_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return data.cvel[base_id][:3]


def compute_base_quat(data):
    base_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return data.xquat[base_id]


def compute_base_pos(data):
    base_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return data.xpos[base_id]


def compute_base_vel(data):
    base_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return data.cvel[base_id][3:]


def compute_com(data):
    """Return total CoM. For robot-only, we approximate as base position."""
    return data.subtree_com[0]


def build_obs(data, joints, commands, last_action, cfg):
    """Build single-timestep obs vector (96 dims)."""
    base_ang_vel = compute_base_ang_vel(data) * np.array(cfg["observations"]["base_ang_vel"]["scale"])
    projected_gravity = compute_projected_gravity(data)
    velocity_commands = np.array(commands, dtype=np.float32)

    default_pos = np.array(cfg["actions"]["JointPositionAction"]["offset"], dtype=np.float32)
    joint_pos = np.array([data.qpos[j["qposadr"]] for j in joints], dtype=np.float32)
    joint_pos_rel = joint_pos - default_pos

    joint_vel_scale = np.array(cfg["observations"]["joint_vel_rel"]["scale"], dtype=np.float32)
    joint_vel = np.array([data.qvel[j["qveladr"]] for j in joints], dtype=np.float32)
    joint_vel_rel = joint_vel * joint_vel_scale

    obs = np.concatenate([
        base_ang_vel,
        projected_gravity,
        velocity_commands,
        joint_pos_rel,
        joint_vel_rel,
        last_action,
    ]).astype(np.float32)
    return obs


def apply_force(data, force_body_id, force, app_point):
    """Apply external force at a point on a body."""
    # Use xfrc_applied at body CoM for simplicity; point application is harder.
    data.xfrc_applied[force_body_id, :3] = force


def is_fallen(data, max_steps, step):
    """Detect fall: base too low or too tilted."""
    base_pos = compute_base_pos(data)
    base_quat = compute_base_quat(data)
    # Convert quaternion to roll/pitch.
    w, x, y, z = base_quat
    pitch = np.arcsin(2 * (w * y - z * x))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    if base_pos[2] < 0.5:
        return True
    if abs(roll) > 0.8 or abs(pitch) > 0.8:
        return True
    if step >= max_steps - 1:
        return False
    return False


def run_episode(seed, cfg, tmp_scene):
    """Run one rollout and return DataFrame."""
    rng = np.random.default_rng(seed)

    slope = 0.0  # flat ground for base policy stability; add slope later if Hema ONNX available
    friction = rng.uniform(0.5, 1.0)
    force_mag = rng.uniform(0.0, 50.0)
    force_body = rng.choice(["pelvis", "torso_link"])
    force_vec = np.array([
        rng.normal(-force_mag * 0.5, force_mag * 0.2),
        rng.normal(0, force_mag * 0.1),
        rng.normal(-force_mag * 0.5, force_mag * 0.2),
    ])
    force_start_step = rng.integers(50, 150)
    force_app_point = np.array([rng.normal(0, 0.05), rng.normal(0, 0.05), rng.normal(1.0, 0.1)])
    commands = np.array([
        rng.uniform(0.3, 0.8),
        rng.uniform(-0.1, 0.1),
        rng.uniform(-0.1, 0.1),
    ], dtype=np.float32)

    make_sloped_scene(G1_XML, slope, friction, tmp_scene)
    model, data = load_model_with_actuators(tmp_scene)
    joints = get_actuated_joint_info(model)

    if len(joints) != 29:
        raise ValueError(f"Expected 29 actuated joints, got {len(joints)}")

    force_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, force_body)

    # ONNX session
    sess = ort.InferenceSession(POLICY_ONNX)
    input_name = sess.get_inputs()[0].name

    # Set initial pose close to default.
    default_pos = np.array(cfg["actions"]["JointPositionAction"]["offset"], dtype=np.float32)
    for j, jp in zip(joints, default_pos):
        data.qpos[j["qposadr"]] = jp

    # Keep MuJoCo physics timestep at 0.002; run 10 substeps per control step.
    physics_dt = model.opt.timestep
    steps_per_ctrl = int(STEP_DT / physics_dt)

    # Warmup stand on flat ground.
    for _ in range(200):
        for j, jp in zip(joints, default_pos):
            data.ctrl[j["ctrlid"]] = jp
        for _ in range(steps_per_ctrl):
            mujoco.mj_step(model, data)

    # History buffer.
    obs_buf = []
    last_action = np.zeros(29, dtype=np.float32)

    rows = []
    fall_step = None

    for t in range(MAX_STEPS):
        obs = build_obs(data, joints, commands, last_action, cfg)
        obs_buf.append(obs)

        if len(obs_buf) >= HISTORY_LEN:
            flat_obs = np.concatenate(obs_buf[-HISTORY_LEN:]).reshape(1, -1)
            action = sess.run(None, {input_name: flat_obs})[0].flatten()
            action = np.clip(action, -1, 1)
            last_action = action.copy()

            # Map policy action to MuJoCo ctrl.
            joint_ids_map = cfg["joint_ids_map"]
            for policy_idx, mj_idx in enumerate(joint_ids_map):
                target = default_pos[policy_idx] + action[policy_idx] * cfg["actions"]["JointPositionAction"]["scale"][policy_idx]
                data.ctrl[joints[mj_idx]["ctrlid"]] = target
        else:
            # During initial warm-up within episode, hold default pose.
            for j, jp in zip(joints, default_pos):
                data.ctrl[j["ctrlid"]] = jp

        if t >= force_start_step:
            apply_force(data, force_body_id, force_vec, force_app_point)
        for _ in range(steps_per_ctrl):
            mujoco.mj_step(model, data)

        # Logging.
        base_pos = compute_base_pos(data)
        base_quat = compute_base_quat(data)
        base_vel = compute_base_vel(data)
        base_ang_vel = compute_base_ang_vel(data)
        projected_gravity = compute_projected_gravity(data)
        com = compute_com(data)

        row = {
            "episode_id": seed,
            "time": t * STEP_DT,
            "timestep": t,
            "slope_angle_deg": slope,
            "friction": friction,
            "base_pos_x": base_pos[0],
            "base_pos_y": base_pos[1],
            "base_pos_z": base_pos[2],
            "base_quat_w": base_quat[0],
            "base_quat_x": base_quat[1],
            "base_quat_y": base_quat[2],
            "base_quat_z": base_quat[3],
            "robot_com_x": base_pos[0],
            "robot_com_y": base_pos[1],
            "robot_com_z": base_pos[2],
            "system_com_x": com[0],
            "system_com_y": com[1],
            "system_com_z": com[2],
            "base_vel_x": base_vel[0],
            "base_vel_y": base_vel[1],
            "base_vel_z": base_vel[2],
            "base_ang_vel_x": base_ang_vel[0],
            "base_ang_vel_y": base_ang_vel[1],
            "base_ang_vel_z": base_ang_vel[2],
            "projected_gravity_x": projected_gravity[0],
            "projected_gravity_y": projected_gravity[1],
            "projected_gravity_z": projected_gravity[2],
            "cmd_vel_x": commands[0],
            "cmd_vel_y": commands[1],
            "cmd_yaw_rate": commands[2],
            "force_mag": force_mag,
            "force_x": force_vec[0],
            "force_y": force_vec[1],
            "force_z": force_vec[2],
            "force_app_x": force_app_point[0],
            "force_app_y": force_app_point[1],
            "force_app_z": force_app_point[2],
        }
        for i, j in enumerate(joints):
            row[f"joint_pos_{i}"] = data.qpos[j["qposadr"]]
            row[f"joint_vel_{i}"] = data.qvel[j["qveladr"]]
            row[f"joint_default_{i}"] = default_pos[i]
            row[f"last_action_{i}"] = last_action[i]

        rows.append(row)

        if is_fallen(data, MAX_STEPS, t) and fall_step is None:
            fall_step = t
            break

    df = pd.DataFrame(rows)
    n = len(df)
    df["fall_label"] = 0
    df["steps_to_fall"] = -1
    if fall_step is not None:
        mask = (df["timestep"] <= fall_step) & (fall_step <= df["timestep"] + PREDICTION_HORIZON)
        df.loc[mask, "fall_label"] = 1
        df.loc[mask, "steps_to_fall"] = fall_step - df.loc[mask, "timestep"]

    return df


def collect_one(args):
    seed, cfg_path, tmp_scene = args
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return run_episode(seed, cfg, tmp_scene)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="/home/hemad/calhacks/data/g1_mujoco_data.csv")
    args = parser.parse_args()

    cfg = load_deploy_config()
    tmp_dir = G1_SCENE_DIR

    print(f"Collecting {args.episodes} episodes with {args.workers} workers...")

    tasks = [(seed, DEPLOY_YAML, os.path.join(tmp_dir, f"scene_{seed}.xml")) for seed in range(args.episodes)]

    if args.workers == 1:
        frames = [collect_one(t) for t in tasks]
    else:
        with Pool(args.workers) as pool:
            frames = pool.map(collect_one, tasks)

    df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Wrote {len(df)} rows from {df['episode_id'].nunique()} episodes to {args.output}")
    print(f"Fall rate: {df['fall_label'].mean():.3f}")


if __name__ == "__main__":
    main()
