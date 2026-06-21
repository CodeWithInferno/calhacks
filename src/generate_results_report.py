"""Generate a beautiful HTML + PNG results report for README/presentation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import yaml
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train_world_model import FallPredictorGRU, FallPredictorMLP, build_windows, load_schema


def evaluate_and_save(model, windows, labels, device, output_csv: Path):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(labels), 256):
            xb = torch.tensor(windows[i : i + 256], dtype=torch.float32).to(device)
            yb = labels[i : i + 256]
            logits = model(xb).cpu().numpy()
            scores = 1.0 / (1.0 + np.exp(-logits))
            all_scores.extend(scores)
            all_labels.extend(yb)
    y_true = np.array(all_labels)
    y_score = np.array(all_scores)
    y_pred = (y_score > 0.5).astype(int)

    pd.DataFrame({"y_true": y_true, "y_score": y_score, "y_pred": y_pred}).to_csv(
        output_csv, index=False
    )

    return {
        "auc": roc_auc_score(y_true, y_score),
        "ap": average_precision_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "y_true": y_true,
        "y_score": y_score,
    }


def _roc_fig(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"GRU (AUC = {auc:.4f})",
            line=dict(color="#6366f1", width=3),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.15)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="#94a3b8", dash="dash", width=2),
        )
    )
    fig.update_layout(
        title="World Model ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        legend=dict(x=0.65, y=0.15),
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def _cm_fig(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Fall", "Fall"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm[::-1],
            x=labels,
            y=labels[::-1],
            text=cm[::-1],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        template="plotly_white",
        margin=dict(l=80, r=40, t=60, b=60),
    )
    return fig


def _risk_dist_fig(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df["max_risk"],
            nbinsx=30,
            marker_color="#ef4444",
            opacity=0.8,
            name="Max Risk",
        )
    )
    fig.update_layout(
        title="Distribution of Peak Fall Risk (Benchmark)",
        xaxis_title="Max Risk Score",
        yaxis_title="Episodes",
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig


def _suite_bar_fig(df: pd.DataFrame):
    summary = (
        df.groupby("suite")
        .agg(
            fall_rate=("fell", "mean"),
            med_dur=("duration_s", "median"),
            mean_risk=("max_risk", "mean"),
        )
        .reset_index()
    )
    order = ["easy_flat", "medium_slope", "hard_slope", "filbert_street", "bradford_street"]
    summary["suite"] = pd.Categorical(summary["suite"], categories=order, ordered=True)
    summary = summary.sort_values("suite")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Fall Rate by Terrain", "Median Survival Time"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Bar(
            x=summary["suite"],
            y=summary["fall_rate"],
            marker_color="#f59e0b",
            name="Fall Rate",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=summary["suite"],
            y=summary["med_dur"],
            marker_color="#10b981",
            name="Median Duration (s)",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=80),
    )
    fig.update_yaxes(title_text="Fall Rate", row=1, col=1)
    fig.update_yaxes(title_text="Seconds", row=1, col=2)
    return fig


def _ppo_progress_fig():
    # Hard-coded progress from the v4 training run (flat, seed 42, 0.5 m/s).
    checkpoints = [0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.5]
    distance = [0.64, 0.69, 1.50, 1.70, 1.77, 1.97, 2.02, 2.22]
    duration = [0.92, 0.88, 1.36, 1.52, 1.58, 1.64, 1.66, 1.74]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Forward Distance", "Survival Time"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=checkpoints,
            y=distance,
            mode="lines+markers",
            line=dict(color="#6366f1", width=3),
            marker=dict(size=8),
            name="Distance (m)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=checkpoints,
            y=duration,
            mode="lines+markers",
            line=dict(color="#10b981", width=3),
            marker=dict(size=8),
            name="Duration (s)",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=60),
    )
    fig.update_xaxes(title_text="Training Steps (M)", row=1, col=1)
    fig.update_xaxes(title_text="Training Steps (M)", row=1, col=2)
    fig.update_yaxes(title_text="Meters", row=1, col=1)
    fig.update_yaxes(title_text="Seconds", row=1, col=2)
    return fig


def generate_html(metrics, y_true, y_score, benchmark_df, output_html: Path):
    roc_html = _roc_fig(y_true, y_score).to_html(full_html=False, include_plotlyjs="cdn")
    cm_html = _cm_fig(y_true, (y_score > 0.5).astype(int)).to_html(
        full_html=False, include_plotlyjs=False
    )
    risk_html = _risk_dist_fig(benchmark_df).to_html(full_html=False, include_plotlyjs=False)
    suite_html = _suite_bar_fig(benchmark_df).to_html(full_html=False, include_plotlyjs=False)
    ppo_html = _ppo_progress_fig().to_html(full_html=False, include_plotlyjs=False)

    cards = f"""
    <div class="grid">
      <div class="card"><div class="num">{metrics['auc']:.4f}</div><div class="label">Test AUC</div></div>
      <div class="card"><div class="num">{metrics['f1']:.4f}</div><div class="label">Test F1</div></div>
      <div class="card"><div class="num">{metrics['accuracy']:.4f}</div><div class="label">Accuracy</div></div>
      <div class="card"><div class="num">{metrics['ap']:.4f}</div><div class="label">Average Precision</div></div>
    </div>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>G1 Risk Copilot — Results</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #1e293b;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #6366f1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 40px 24px;
    }}
    header {{
      text-align: center;
      margin-bottom: 48px;
    }}
    h1 {{
      font-size: 2.6rem;
      margin: 0 0 8px;
      background: linear-gradient(90deg, #818cf8, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    p.subtitle {{
      color: var(--muted);
      font-size: 1.1rem;
      margin: 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 20px;
      margin-bottom: 40px;
    }}
    .card {{
      background: var(--panel);
      border-radius: 16px;
      padding: 24px;
      text-align: center;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }}
    .card .num {{
      font-size: 2.4rem;
      font-weight: 700;
      color: var(--accent);
    }}
    .card .label {{
      color: var(--muted);
      font-size: 0.95rem;
      margin-top: 4px;
    }}
    .section {{
      background: var(--panel);
      border-radius: 20px;
      padding: 28px;
      margin-bottom: 28px;
      box-shadow: 0 10px 15px -3px rgba(0,0,0,0.2);
    }}
    .section h2 {{
      margin-top: 0;
      font-size: 1.4rem;
      color: #e2e8f0;
    }}
    .note {{
      color: var(--muted);
      font-size: 0.95rem;
      margin-top: 12px;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }}
    @media (max-width: 800px) {{
      .two-col {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>G1 Load & Slope Risk Copilot</h1>
      <p class="subtitle">MuJoCo simulation · GRU world model · PPO walking policy</p>
    </header>

    {cards}

    <div class="section">
      <h2>World Model Performance</h2>
      <div class="two-col">
        {roc_html}
        {cm_html}
      </div>
      <p class="note">
        The GRU classifier is trained on episode-split data and evaluated on a held-out test set.
        It predicts fall risk from a 10-timestep window of state, slope, and force features.
      </p>
    </div>

    <div class="section">
      <h2>Controller Benchmark (Heuristic vs Safe-MPC)</h2>
      <div class="two-col">
        {suite_html}
        {risk_html}
      </div>
      <p class="note">
        Both controllers fall on every tested terrain, but the world model assigns near-maximum risk,
        demonstrating that the critic is working even when the controller is the bottleneck.
      </p>
    </div>

    <div class="section">
      <h2>PPO Walking Policy Training Progress</h2>
      {ppo_html}
      <p class="note">
        PPO v4 is trained directly in MuJoCo. Each checkpoint is evaluated on flat ground (seed 42, 0.5 m/s).
        Distance and survival time are steadily improving as training continues toward 5M steps.
      </p>
    </div>
  </div>
</body>
</html>
"""
    output_html.write_text(html)


def screenshot(html_path: Path, png_path: Path):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 1600})
        page.goto(f"file://{html_path.resolve()}")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(png_path), full_page=True)
        browser.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_valloss.yaml")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg["data_path"])
    schema = load_schema()
    feature_cols = schema["feature_cols"]
    episode_id_col = schema["episode_id_col"]
    label_col = schema["label_col"]

    episode_ids = df[episode_id_col].unique()
    train_eps, test_eps = train_test_split(episode_ids, test_size=0.15, random_state=42)
    val_frac = cfg.get("val_frac", 0.1)
    train_eps, val_eps = train_test_split(
        train_eps, test_size=val_frac / (1 - 0.15), random_state=42
    )

    train_df = df[df[episode_id_col].isin(train_eps)].copy()
    val_df = df[df[episode_id_col].isin(val_eps)].copy()
    test_df = df[df[episode_id_col].isin(test_eps)].copy()

    model_dir = Path(cfg["output_dir"])
    means = pd.read_csv(model_dir / "feature_means.csv", index_col=0)["0"]
    stds = pd.read_csv(model_dir / "feature_stds.csv", index_col=0)["0"]

    for d in (train_df, val_df, test_df):
        d[feature_cols] = (d[feature_cols] - means) / stds

    window_size = cfg["window_size"]
    X_train, y_train = build_windows(train_df, feature_cols, episode_id_col, label_col, window_size)
    X_val, y_val = build_windows(val_df, feature_cols, episode_id_col, label_col, window_size)
    X_test, y_test = build_windows(test_df, feature_cols, episode_id_col, label_col, window_size)

    input_dim = len(feature_cols)
    if cfg["model_type"] == "gru":
        model = FallPredictorGRU(input_dim, cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"])
    else:
        model = FallPredictorMLP(input_dim * window_size, cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model = model.to(device)

    preds_csv = output_dir / "world_model_predictions.csv"
    metrics = evaluate_and_save(model, X_test, y_test, device, preds_csv)
    print("World model metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"  {k}: {v:.4f}")

    benchmark_csv = output_dir / "benchmark" / "benchmark_runs.csv"
    benchmark_df = pd.read_csv(benchmark_csv) if benchmark_csv.exists() else pd.DataFrame()

    html_path = output_dir / "results_report.html"
    png_path = output_dir / "results_report.png"
    generate_html(metrics, metrics["y_true"], metrics["y_score"], benchmark_df, html_path)
    screenshot(html_path, png_path)

    print(f"Report: {html_path}")
    print(f"Screenshot: {png_path}")


if __name__ == "__main__":
    main()
