"""Train a PPO walking policy for G1 in MuJoCo."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from g1_walk_env import G1WalkEnv


def make_env(slope_deg: float, friction: float, seed: int = 0):
    def _init():
        env = G1WalkEnv(
            slope_deg=slope_deg,
            friction=friction,
        )
        env.reset(seed=seed)
        env = Monitor(env)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/g1_ppo_walk_v2")
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--slope-deg", type=float, default=0.0)
    parser.add_argument("--friction", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [
        make_env(args.slope_deg, args.friction, seed=i)
        for i in range(args.n_envs)
    ]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=[256, 128]),
        device="auto",
    )

    # save_freq is per-env steps; divide so total checkpoint interval is 250k.
    checkpoint_callback = CheckpointCallback(
        save_freq=250_000 // args.n_envs,
        save_path=str(output_dir),
        name_prefix="g1_ppo",
    )

    start = time.time()
    model.learn(total_timesteps=args.steps, callback=checkpoint_callback)
    model.save(output_dir / "g1_ppo_final")
    print(f"Training complete in {(time.time() - start)/3600:.2f}h")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
