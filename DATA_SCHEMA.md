# Data schema for G1 fall-risk dataset

One row per simulation timestep.

## Identifiers

- `episode_id` — rollout number
- `time` — simulation time in seconds


No separate t-1/t/t+1 columns needed. The training script builds sliding windows from rows ordered by `timestep`.

## Terrain

- `slope_angle_deg` — ground slope in degrees
- `friction` — floor friction coefficient

## Base state

- `base_pos_x`, `base_pos_y`, `base_pos_z` — base position
- `base_quat_w`, `base_quat_x`, `base_quat_y`, `base_quat_z` — base orientation
- `robot_com_x`, `robot_com_y`, `robot_com_z` — robot-only center of mass
- `system_com_x`, `system_com_y`, `system_com_z` — robot + external load combined center of mass
- `base_vel_x`, `base_vel_y`, `base_vel_z` — linear velocity
- `base_ang_vel_x`, `base_ang_vel_y`, `base_ang_vel_z` — angular velocity
- `projected_gravity_x`, `projected_gravity_y`, `projected_gravity_z` — gravity in base frame
- Center of gravity

## Joint state (29 DoF)

- `joint_pos_0` ... `joint_pos_28` — joint positions
- `joint_vel_0` ... `joint_vel_28` — joint velocities
- `joint_default_0` ... `joint_default_28` — default standing positions

## Commands / actions

- `cmd_vel_x`, `cmd_vel_y`, `cmd_yaw_rate` — velocity command
- `last_action_0` ... `last_action_28` — policy output from previous step

## External load / force

- `force_mag` — force magnitude in Newtons
- `force_x`, `force_y`, `force_z` — force vector in world frame
- `force_body` — body name where force is applied
- `force_app_x`, `force_app_y`, `force_app_z` — application point in world frame

## Labels

- `fall_label` — `1` if robot falls within next 25 timesteps (0.5 s) from this row, else `0`
- `steps_to_fall` — (optional) number of timesteps until fall; `-1` if no fall in this episode
