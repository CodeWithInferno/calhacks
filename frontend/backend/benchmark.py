"""Production-style benchmark for the G1 fall-prediction system.

Runs multiple test suites (Easy / Medium / Hard / Real-world SF hills) across
controllers (heuristic, safe-MPC) and records robustness + world-model
performance metrics.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mujoco_collector.collector import EpisodeConfig
from mujoco_rollout import MUJOCO_AVAILABLE, MUJOCO_ERROR, run_episode_controller
from risk_model import score_dataframe


@dataclass
class SuiteConfig:
    name: str
    n_episodes: int
    slope_deg: tuple[float, float]
    friction: tuple[float, float]
    speed_mps: tuple[float, float]
    lateral_mps: tuple[float, float] = (0.0, 0.0)
    yaw_rate: tuple[float, float] = (0.0, 0.0)
    push_max: float = 0.0
    description: str = ""


SUITES = [
    SuiteConfig(
        name="easy_flat",
        n_episodes=50,
        slope_deg=(0.0, 3.0),
        friction=(0.9, 1.0),
        speed_mps=(0.3, 0.6),
        description="Flat ground, high friction, slow walking.",
    ),
    SuiteConfig(
        name="medium_slope",
        n_episodes=50,
        slope_deg=(5.0, 15.0),
        friction=(0.6, 0.85),
        speed_mps=(0.5, 1.0),
        push_max=30.0,
        description="Moderate slopes, varied friction, occasional pushes.",
    ),
    SuiteConfig(
        name="hard_slope",
        n_episodes=50,
        slope_deg=(20.0, 30.0),
        friction=(0.4, 0.65),
        speed_mps=(0.8, 1.5),
        push_max=80.0,
        description="Steep slopes, low friction, aggressive commands and pushes.",
    ),
    SuiteConfig(
        name="filbert_street",
        n_episodes=50,
        slope_deg=(17.4, 17.4),  # 31.5% grade
        friction=(0.5, 0.9),
        speed_mps=(0.3, 0.8),
        push_max=20.0,
        description="Real SF: Filbert St, 31.5% grade.",
    ),
    SuiteConfig(
        name="bradford_street",
        n_episodes=50,
        slope_deg=(22.3, 22.3),  # 41% grade
        friction=(0.5, 0.9),
        speed_mps=(0.2, 0.6),
        push_max=10.0,
        description="Real SF: Bradford above Tompkins, 41% grade.",
    ),
]

CONTROLLERS = ["heuristic", "safe"]


@dataclass
class EpisodeResult:
    suite: str
    controller: str
    episode: int
    seed: int
    slope_deg: float
    friction: float
    speed_mps: float
    lateral_mps: float
    yaw_rate: float
    push_mag: float
    fell: int
    duration_s: float
    distance_m: float
    final_height_m: float
    max_risk: float
    mean_risk: float
    steps: int


def _sample_suite(suite: SuiteConfig, episode: int, base_seed: int) -> tuple[EpisodeConfig, dict]:
    rng = np.random.default_rng(base_seed + episode)
    slope_deg = float(rng.uniform(*suite.slope_deg))
    friction = float(rng.uniform(*suite.friction))
    speed = float(rng.uniform(*suite.speed_mps))
    lateral = float(rng.uniform(*suite.lateral_mps))
    yaw = float(rng.uniform(*suite.yaw_rate))
    push_mag = float(rng.uniform(0.0, suite.push_max))
    push_dir = rng.normal(size=3).astype(np.float32)
    push_dir /= np.linalg.norm(push_dir) + 1e-8
    push = (push_dir * push_mag).astype(np.float32)

    cfg = EpisodeConfig(
        episode_id=episode,
        seed=base_seed + episode,
        max_steps=300,
        slope_deg=slope_deg,
        friction=friction,
        cmd_vel=np.array([speed, lateral], dtype=np.float32),
        cmd_yaw_rate=yaw,
        force=push,
        force_body="pelvis",
    )
    meta = {
        "slope_deg": slope_deg,
        "friction": friction,
        "speed_mps": speed,
        "lateral_mps": lateral,
        "yaw_rate": yaw,
        "push_mag": push_mag,
    }
    return cfg, meta


def _is_fallen(df: pd.DataFrame) -> tuple[bool, float]:
    if len(df) == 0:
        return True, 0.0
    last = df.iloc[-1]
    # max_steps reached -> did not fall within window.
    fell = len(df) < 300
    return fell, float(last["time"])


def run_benchmark(output_dir: Path, controllers: list[str] | None = None):
    if not MUJOCO_AVAILABLE:
        raise RuntimeError(f"MuJoCo not available: {MUJOCO_ERROR}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    controllers = controllers or CONTROLLERS
    results: list[EpisodeResult] = []
    start_total = time.time()

    for suite in SUITES:
        print(f"\n=== Suite: {suite.name} ({suite.description}) ===")
        for controller in controllers:
            suite_start = time.time()
            suite_results = []
            for ep in range(suite.n_episodes):
                cfg, meta = _sample_suite(suite, ep, base_seed=abs(hash(suite.name)) % 1_000_000)
                frames, df = run_episode_controller(cfg, controller)
                fell, duration = _is_fallen(df)
                distance = float(df["base_pos_x"].iloc[-1]) if len(df) else 0.0
                final_height = float(df["base_pos_z"].iloc[-1]) if len(df) else 0.0

                # Score risk.
                scores = score_dataframe(df)
                max_risk = float(scores.max()) if len(scores.dropna()) else 0.0
                mean_risk = float(scores.mean()) if len(scores.dropna()) else 0.0

                res = EpisodeResult(
                    suite=suite.name,
                    controller=controller,
                    episode=ep,
                    seed=cfg.seed,
                    slope_deg=meta["slope_deg"],
                    friction=meta["friction"],
                    speed_mps=meta["speed_mps"],
                    lateral_mps=meta["lateral_mps"],
                    yaw_rate=meta["yaw_rate"],
                    push_mag=meta["push_mag"],
                    fell=int(fell),
                    duration_s=duration,
                    distance_m=distance,
                    final_height_m=final_height,
                    max_risk=max_risk,
                    mean_risk=mean_risk,
                    steps=len(df),
                )
                results.append(res)
                suite_results.append(res)

            n = len(suite_results)
            fall_rate = sum(r.fell for r in suite_results) / n
            med_dur = float(np.median([r.duration_s for r in suite_results]))
            med_dist = float(np.median([r.distance_m for r in suite_results]))
            print(
                f"  {controller:10s}: fall_rate={fall_rate:.2%} "
                f"median_duration={med_dur:.2f}s median_dist={med_dist:.3f}m "
                f"({time.time() - suite_start:.1f}s)"
            )

    df = pd.DataFrame([asdict(r) for r in results])
    csv_path = output_dir / "benchmark_runs.csv"
    df.to_csv(csv_path, index=False)

    summary = _summarize(df)
    json_path = output_dir / "benchmark_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"\nBenchmark complete in {time.time() - start_total:.1f}s")
    print(f"Results: {csv_path}")
    print(f"Summary: {json_path}")
    return df, summary


def _summarize(df: pd.DataFrame) -> dict:
    summary = {}
    overall = {
        "n_runs": int(len(df)),
        "fall_rate": float(df["fell"].mean()),
        "median_duration_s": float(df["duration_s"].median()),
        "median_distance_m": float(df["distance_m"].median()),
    }
    # Risk calibration / AUC.
    valid = df.dropna(subset=["max_risk"])
    if len(valid) > 0 and valid["fell"].nunique() > 1:
        overall["risk_auc"] = float(roc_auc_score(valid["fell"], valid["max_risk"]))
    summary["overall"] = overall

    by_suite_controller = (
        df.groupby(["suite", "controller"])
        .agg(
            n_runs=("fell", "count"),
            fall_rate=("fell", "mean"),
            median_duration_s=("duration_s", "median"),
            median_distance_m=("distance_m", "median"),
            mean_max_risk=("max_risk", "mean"),
        )
        .reset_index()
    )
    summary["by_suite_controller"] = by_suite_controller.to_dict(orient="records")

    # Head-to-head controller comparison per suite.
    summary["controller_comparison"] = {}
    for suite in df["suite"].unique():
        sub = df[df["suite"] == suite]
        comp = {}
        for controller in sub["controller"].unique():
            csub = sub[sub["controller"] == controller]
            comp[controller] = {
                "fall_rate": float(csub["fell"].mean()),
                "median_duration_s": float(csub["duration_s"].median()),
                "median_distance_m": float(csub["distance_m"].median()),
            }
        summary["controller_comparison"][suite] = comp

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/home/hemad/calhacks/results/benchmark")
    parser.add_argument("--controllers", nargs="+", default=None, choices=CONTROLLERS)
    args = parser.parse_args()
    run_benchmark(Path(args.output), args.controllers)
