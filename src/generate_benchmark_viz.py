"""Generate a polished benchmark summary image for README/presentation."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).parent.parent


def _add_card(ax, x, y, width, height, value, label, color):
    rect = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#1e293b",
        edgecolor="#334155",
        linewidth=1.5,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)
    ax.text(
        x + width / 2,
        y + height * 0.55,
        value,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=32,
        fontweight="bold",
        color=color,
    )
    ax.text(
        x + width / 2,
        y + height * 0.18,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="#94a3b8",
    )


def generate(output_path: Path):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 10), facecolor="#0f172a")
    fig.patch.set_facecolor("#0f172a")

    # Title area
    ax_title = fig.add_axes([0.0, 0.88, 1.0, 0.12])
    ax_title.set_facecolor("#0f172a")
    ax_title.axis("off")
    ax_title.text(
        0.5,
        0.55,
        "G1 Risk Copilot — Benchmark",
        ha="center",
        va="center",
        fontsize=32,
        fontweight="bold",
        color="#f8fafc",
        transform=ax_title.transAxes,
    )
    ax_title.text(
        0.5,
        0.18,
        "GRU world model + PPO walking policy trained in MuJoCo",
        ha="center",
        va="center",
        fontsize=14,
        color="#94a3b8",
        transform=ax_title.transAxes,
    )

    # Metric cards
    ax_cards = fig.add_axes([0.05, 0.70, 0.9, 0.16])
    ax_cards.set_facecolor("#0f172a")
    ax_cards.axis("off")
    _add_card(ax_cards, 0.02, 0.1, 0.22, 0.8, "0.9971", "World Model AUC", "#818cf8")
    _add_card(ax_cards, 0.27, 0.1, 0.22, 0.8, "97.4%", "Fall Prediction Accuracy", "#34d399")
    _add_card(ax_cards, 0.52, 0.1, 0.22, 0.8, "3.34 m", "PPO v4 Forward Distance", "#f472b6")
    _add_card(ax_cards, 0.77, 0.1, 0.22, 0.8, "2.54 s", "PPO v4 Survival Time", "#fbbf24")

    # Controller comparison (flat ground, seed 42, 0.5 m/s)
    ax_bar = fig.add_axes([0.07, 0.38, 0.55, 0.28])
    ax_bar.set_facecolor("#0f172a")
    controllers = ["Heuristic", "Safe-MPC", "PPO v4"]
    distance = [0.60, 0.59, 3.34]
    survival = [0.88, 0.89, 2.54]
    x = np.arange(len(controllers))
    width = 0.35
    bars1 = ax_bar.bar(x - width / 2, distance, width, label="Distance (m)", color="#6366f1", edgecolor="#4f46e5")
    bars2 = ax_bar.bar(x + width / 2, survival, width, label="Survival (s)", color="#10b981", edgecolor="#059669")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(controllers, fontsize=13, color="#e2e8f0")
    ax_bar.set_ylabel("Value", fontsize=12, color="#94a3b8")
    ax_bar.set_title("Controller Comparison — Flat Ground", fontsize=16, fontweight="bold", color="#f8fafc", pad=12)
    ax_bar.legend(frameon=False, labelcolor="#e2e8f0", fontsize=11)
    ax_bar.tick_params(colors="#94a3b8")
    for spine in ax_bar.spines.values():
        spine.set_color("#334155")
    ax_bar.set_ylim(0, 3.8)
    for bar in bars1:
        height = bar.get_height()
        ax_bar.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=10, color="#e2e8f0")
    for bar in bars2:
        height = bar.get_height()
        ax_bar.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=10, color="#e2e8f0")

    # Terrain difficulty overview
    ax_terrain = fig.add_axes([0.68, 0.38, 0.27, 0.28])
    ax_terrain.set_facecolor("#0f172a")
    suites = ["Easy Flat", "Medium\nSlope", "Hard\nSlope", "Filbert St", "Bradford St"]
    grades = [0, 10, 25, 17.4, 22.3]
    colors = ["#34d399", "#fbbf24", "#f97316", "#ef4444", "#b91c1c"]
    bars = ax_terrain.barh(suites, grades, color=colors, edgecolor="white", linewidth=0.5)
    ax_terrain.set_xlabel("Slope (°)", fontsize=11, color="#94a3b8")
    ax_terrain.set_title("Benchmark Terrain Grades", fontsize=16, fontweight="bold", color="#f8fafc", pad=12)
    ax_terrain.tick_params(colors="#94a3b8")
    for spine in ax_terrain.spines.values():
        spine.set_color("#334155")
    ax_terrain.invert_yaxis()
    for bar, grade in zip(bars, grades):
        ax_terrain.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{grade:.1f}°", va="center", ha="left", fontsize=10, color="#e2e8f0")

    # Risk model calibration text
    ax_text = fig.add_axes([0.07, 0.08, 0.88, 0.24])
    ax_text.set_facecolor("#1e293b")
    ax_text.axis("off")
    ax_text.text(
        0.5,
        0.88,
        "Why this matters",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#f8fafc",
        transform=ax_text.transAxes,
    )
    bullets = [
        "• The GRU world model predicts falls with 99.71% AUC on held-out test episodes.",
        "• Heuristic and Safe-MPC controllers fall on every tested terrain — the controller is the bottleneck, not the critic.",
        "• PPO v4 is trained directly in MuJoCo and walks 3.3 m on flat ground after 5M steps — 5.6× farther than the heuristic.",
        "• The backend serves real-time MuJoCo rollouts with per-frame risk scores to the frontend viewer.",
    ]
    for i, b in enumerate(bullets):
        ax_text.text(
            0.03,
            0.62 - i * 0.16,
            b,
            ha="left",
            va="top",
            fontsize=12,
            color="#cbd5e1",
            transform=ax_text.transAxes,
        )

    # Footer
    ax_footer = fig.add_axes([0.0, 0.0, 1.0, 0.05])
    ax_footer.set_facecolor("#0f172a")
    ax_footer.axis("off")
    ax_footer.text(
        0.5,
        0.5,
        "CalHacks 2026 · github.com/CodeWithInferno/calhacks",
        ha="center",
        va="center",
        fontsize=10,
        color="#64748b",
        transform=ax_footer.transAxes,
    )

    fig.savefig(output_path, dpi=150, facecolor="#0f172a", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved benchmark viz to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/benchmark_summary.png")
    args = parser.parse_args()
    generate(Path(args.output))


if __name__ == "__main__":
    main()
