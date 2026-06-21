"""Load the trained fall-risk GRU and score a rollout DataFrame."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

_MODEL_DIR = Path(__file__).parent / "model" / "mujoco_valloss"


class FallPredictorGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


def _load_artifacts():
    with open(_MODEL_DIR / "mujoco_valloss.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(_MODEL_DIR / "schema.yaml") as f:
        schema = yaml.safe_load(f)
    means = pd.read_csv(_MODEL_DIR / "feature_means.csv", index_col=0)["0"]
    stds = pd.read_csv(_MODEL_DIR / "feature_stds.csv", index_col=0)["0"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FallPredictorGRU(
        len(schema["feature_cols"]),
        cfg["hidden_dim"],
        cfg["num_layers"],
        cfg["dropout"],
    )
    model.load_state_dict(torch.load(_MODEL_DIR / "best_model.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return cfg, schema, means, stds, model, device


_CFG, _SCHEMA, _MEANS, _STDS, _MODEL, _DEVICE = _load_artifacts()
_WINDOW = int(_CFG["window_size"])
_FEATURE_COLS = list(_SCHEMA["feature_cols"])


def _build_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    labels = []
    for _, ep_df in df.groupby("episode_id", sort=False):
        features = ep_df[_FEATURE_COLS].to_numpy(dtype=np.float32)
        n = len(features)
        if n <= _WINDOW:
            continue
        shape = (n - _WINDOW, _WINDOW, features.shape[1])
        ws = np.lib.stride_tricks.as_strided(
            features,
            shape=shape,
            strides=(features.strides[0], features.strides[0], features.strides[1]),
        )
        windows.append(ws)
        labels.append(np.zeros(n - _WINDOW, dtype=np.float32))
    if not windows:
        return np.empty((0, _WINDOW, len(_FEATURE_COLS)), dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.concatenate(windows), np.concatenate(labels)


def score_dataframe(df: pd.DataFrame) -> pd.Series:
    """Return a risk score (0-1) for every row that has a valid window ending at it."""
    df = df.copy()
    if "episode_id" not in df.columns:
        df["episode_id"] = 0

    df[_FEATURE_COLS] = (df[_FEATURE_COLS] - _MEANS) / _STDS
    X, _ = _build_windows(df)
    if len(X) == 0:
        return pd.Series(dtype=float)

    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32, device=_DEVICE)
        scores = torch.sigmoid(_MODEL(tensor)).cpu().numpy()

    # Align scores to the last timestep of each window.
    out = pd.Series(index=df.index, dtype=float)
    idx = 0
    for _, ep_df in df.groupby("episode_id", sort=False):
        n = len(ep_df)
        if n > _WINDOW:
            out.iloc[_WINDOW:n] = scores[idx : idx + n - _WINDOW]
            idx += n - _WINDOW
    return out
