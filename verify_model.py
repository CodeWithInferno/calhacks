"""Independent verification of a trained world model on the held-out test set."""
import os
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

from src.train_world_model import FallPredictorGRU, FallPredictorMLP, load_schema, build_windows


def evaluate(model, windows, labels, device):
    model.eval()
    all_scores = []
    all_labels = []
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
    return {
        "auc": roc_auc_score(y_true, y_score),
        "ap": average_precision_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def verify(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg["data_path"])
    schema = load_schema()
    feature_cols = schema["feature_cols"]
    episode_id_col = schema["episode_id_col"]
    label_col = schema["label_col"]

    # Exact same episode split used during training.
    episode_ids = df[episode_id_col].unique()
    train_eps, test_eps = train_test_split(episode_ids, test_size=0.15, random_state=42)
    val_frac = cfg.get("val_frac", 0.1)
    train_eps, val_eps = train_test_split(
        train_eps, test_size=val_frac / (1 - 0.15), random_state=42
    )

    train_df = df[df[episode_id_col].isin(train_eps)].copy()
    val_df = df[df[episode_id_col].isin(val_eps)].copy()
    test_df = df[df[episode_id_col].isin(test_eps)].copy()

    means = pd.read_csv(os.path.join(cfg["output_dir"], "feature_means.csv"), index_col=0)["0"]
    stds = pd.read_csv(os.path.join(cfg["output_dir"], "feature_stds.csv"), index_col=0)["0"]

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
    model.load_state_dict(torch.load(os.path.join(cfg["output_dir"], "best_model.pt"), map_location=device))
    model = model.to(device)

    print(f"Model: {cfg['output_dir']}")
    print(f"Episodes -> train: {len(train_eps)} | val: {len(val_eps)} | test: {len(test_eps)}")
    print(f"Test windows: {len(y_test)} (fall rate {y_test.mean():.3f})")
    print("Independent test metrics:")
    metrics = evaluate(model, X_test, y_test, device)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_valloss.yaml")
    args = parser.parse_args()
    verify(args.config)
