"""
Train a tiny world model to predict fall probability on slope + load data.
"""

import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score


class FallDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


class FallPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FallPredictorGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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

    for _, ep_df in df.groupby(episode_id_col):
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


def train(cfg):
    data_path = cfg["data_path"]
    window_size = cfg["window_size"]
    hidden_dim = cfg["hidden_dim"]
    model_type = cfg["model_type"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["learning_rate"]
    output_dir = cfg["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    schema = load_schema()
    feature_cols = schema["feature_cols"]
    episode_id_col = schema["episode_id_col"]
    label_col = schema["label_col"]

    df = pd.read_csv(data_path)

    # Ensure required columns exist.
    missing = [c for c in feature_cols + [episode_id_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    feature_means = df[feature_cols].mean()
    feature_stds = df[feature_cols].std().replace(0, 1.0)
    df[feature_cols] = (df[feature_cols] - feature_means) / feature_stds

    feature_means.to_csv(os.path.join(output_dir, "feature_means.csv"))
    feature_stds.to_csv(os.path.join(output_dir, "feature_stds.csv"))

    # Save schema for inference.
    with open(os.path.join(output_dir, "schema.yaml"), "w") as f:
        yaml.dump(schema, f)

    print(f"Building windows of size {window_size}...")
    windows, labels = build_windows(df, feature_cols, episode_id_col, label_col, window_size)
    print(f"Total windows: {len(labels)}, fall rate: {labels.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        windows, labels, test_size=0.2, random_state=42, stratify=labels
    )

    input_dim = len(feature_cols)

    if model_type == "mlp":
        model = FallPredictorMLP(input_dim * window_size, hidden_dim)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    elif model_type == "gru":
        model = FallPredictorGRU(input_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    train_ds = FallDataset(X_train, y_train)
    test_ds = FallDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            if torch.isnan(logits).any():
                raise RuntimeError("NaN logits detected")
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb).cpu().numpy()
                pred = 1.0 / (1.0 + np.exp(-logits))
                all_preds.extend(pred)
                all_labels.extend(yb.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds > 0.5)
        avg_loss = total_loss / len(train_loader.dataset)

        print(
            f"Epoch {epoch+1}/{epochs} | loss: {avg_loss:.4f} | test AUC: {auc:.4f} | test acc: {acc:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

    print(f"Best test AUC: {best_auc:.4f}")
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    with open("src/config.yaml") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
