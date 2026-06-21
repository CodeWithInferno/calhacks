"""
Train a world model to predict fall probability.
Includes checkpointing, CSV logging, metrics, early stopping, LR scheduler.
"""

import os
import time
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class FallDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


class FallPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FallPredictorGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
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


def load_schema(path="src/schema.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_windows(df, feature_cols, episode_id_col, label_col, window_size=10):
    windows = []
    labels = []

    for _, ep_df in df.groupby(episode_id_col, sort=False):
        features = ep_df[feature_cols].to_numpy(dtype=np.float32)
        lbls = ep_df[label_col].to_numpy(dtype=np.float32)
        n = len(features)
        if n <= window_size:
            continue

        shape = (n - window_size, window_size, features.shape[1])
        ws = np.lib.stride_tricks.as_strided(
            features,
            shape=shape,
            strides=(features.strides[0], features.strides[0], features.strides[1]),
        )
        windows.append(ws)
        labels.append(lbls[window_size:])

    return np.concatenate(windows), np.concatenate(labels)


def compute_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score > threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_score),
        "ap": average_precision_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_model(model, loader, device):
    model.eval()
    all_scores = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(yb.cpu().numpy())

    y_true = np.array(all_labels)
    y_score = np.array(all_scores)
    metrics = compute_metrics(y_true, y_score)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train(cfg):
    data_path = cfg["data_path"]
    window_size = cfg["window_size"]
    model_type = cfg["model_type"]
    hidden_dim = cfg["hidden_dim"]
    num_layers = cfg["num_layers"]
    dropout = cfg["dropout"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["learning_rate"]
    weight_decay = cfg.get("weight_decay", 0.0)
    output_dir = cfg["output_dir"]
    patience = cfg.get("patience", 10)
    val_frac = cfg.get("val_frac", 0.1)
    save_every = cfg.get("save_every", 1)

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.csv")

    schema = load_schema()
    feature_cols = schema["feature_cols"]
    episode_id_col = schema["episode_id_col"]
    label_col = schema["label_col"]

    df = pd.read_csv(data_path)
    missing = [c for c in feature_cols + [episode_id_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Save artifacts.
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)
    with open(os.path.join(output_dir, "schema.yaml"), "w") as f:
        yaml.dump(schema, f)

    # Split episodes first to avoid windows from the same episode leaking across splits.
    episode_ids = df[episode_id_col].unique()
    train_eps, test_eps = train_test_split(episode_ids, test_size=0.15, random_state=42)
    if val_frac > 0:
        train_eps, val_eps = train_test_split(
            train_eps,
            test_size=val_frac / (1 - 0.15),
            random_state=42,
        )
    else:
        val_eps = test_eps

    train_df = df[df[episode_id_col].isin(train_eps)].copy()
    val_df = df[df[episode_id_col].isin(val_eps)].copy()
    test_df = df[df[episode_id_col].isin(test_eps)].copy()

    print(f"Episodes -> train: {len(train_eps)} | val: {len(val_eps)} | test: {len(test_eps)}")

    # Normalize using training statistics only.
    feature_means = train_df[feature_cols].mean()
    feature_stds = train_df[feature_cols].std()
    # Near-zero stds (effectively constant features) would blow up normalization and
    # make saved stats irreproducible due to floating-point round-trip. Treat them as 1.0.
    std_floor = 1e-12
    feature_stds = feature_stds.where(feature_stds.abs() >= std_floor, 1.0)
    train_df[feature_cols] = (train_df[feature_cols] - feature_means) / feature_stds
    val_df[feature_cols] = (val_df[feature_cols] - feature_means) / feature_stds
    test_df[feature_cols] = (test_df[feature_cols] - feature_means) / feature_stds
    feature_means.to_csv(os.path.join(output_dir, "feature_means.csv"))
    feature_stds.to_csv(os.path.join(output_dir, "feature_stds.csv"))

    print(f"Building windows of size {window_size}...")
    X_train, y_train = build_windows(train_df, feature_cols, episode_id_col, label_col, window_size)
    X_val, y_val = build_windows(val_df, feature_cols, episode_id_col, label_col, window_size)
    X_test, y_test = build_windows(test_df, feature_cols, episode_id_col, label_col, window_size)
    print(
        f"Windows -> train: {len(y_train)} (fall {y_train.mean():.3f}) | "
        f"val: {len(y_val)} (fall {y_val.mean():.3f}) | "
        f"test: {len(y_test)} (fall {y_test.mean():.3f})"
    )

    input_dim = len(feature_cols)

    if model_type == "mlp":
        model = FallPredictorMLP(input_dim * window_size, hidden_dim, num_layers, dropout)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    elif model_type == "gru":
        model = FallPredictorGRU(input_dim, hidden_dim, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type} | params: {total_params:,} | layers: {num_layers} | hidden: {hidden_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Training on {device}")
    model = model.to(device)

    train_ds = FallDataset(X_train, y_train)
    val_ds = FallDataset(X_val, y_val)
    test_ds = FallDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience // 2
    )

    # Logging.
    log_columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_auc",
        "val_ap",
        "val_acc",
        "val_f1",
        "lr",
        "epoch_time",
    ]
    with open(log_path, "w") as f:
        f.write(",".join(log_columns) + "\n")

    best_auc = -1.0
    best_val_loss = float("inf")
    best_metric_value = None
    epochs_no_improve = 0
    start_epoch = 1

    # Resume from checkpoint if exists.
    checkpoint_path = os.path.join(output_dir, "last_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("best_auc", -1.0)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)

        train_loss = total_loss / len(train_loader.dataset)
        val_metrics = evaluate_model(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - t0
        log_row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_auc": f"{val_metrics['auc']:.4f}",
            "val_ap": f"{val_metrics['ap']:.4f}",
            "val_acc": f"{val_metrics['accuracy']:.4f}",
            "val_f1": f"{val_metrics['f1']:.4f}",
            "lr": f"{lr_now:.6f}",
            "epoch_time": f"{epoch_time:.2f}",
        }
        with open(log_path, "a") as f:
            f.write(",".join(str(log_row[c]) for c in log_columns) + "\n")

        print(
            f"Epoch {epoch:3d}/{epochs} | train_loss: {train_loss:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | val_auc: {val_metrics['auc']:.4f} | "
            f"val_f1: {val_metrics['f1']:.4f} | lr: {lr_now:.6f} | time: {epoch_time:.1f}s"
        )

        # Choose early-stopping / checkpointing metric.
        best_metric = cfg.get("best_metric", "val_auc")
        if best_metric == "val_loss":
            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                best_metric_value = best_val_loss
        else:
            is_best = val_metrics["auc"] > best_auc
            if is_best:
                best_auc = val_metrics["auc"]
                best_metric_value = best_auc

        scheduler.step(best_metric_value if best_metric_value is not None else val_metrics["auc"])

        if is_best:
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        else:
            epochs_no_improve += 1

        if epoch % save_every == 0 or is_best:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Final test eval using the best checkpoint (not the last epoch).
    best_model_path = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model = model.to(device)
    test_metrics = evaluate_model(model, test_loader, device)
    print("\nFinal test metrics (best checkpoint):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save final checkpoint.
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc,
            "test_metrics": test_metrics,
        },
        os.path.join(output_dir, "final_checkpoint.pt"),
    )

    if best_metric == "val_loss":
        print(f"\nBest val loss: {best_val_loss:.4f}")
    else:
        print(f"\nBest val AUC: {best_auc:.4f}")
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
