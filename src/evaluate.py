"""
Evaluate a trained world model on a labeled CSV.
Prints AUC, accuracy, precision, recall, F1.
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate(csv_path: str, threshold: float = 0.5):
    df = pd.read_csv(csv_path)
    if "fall_risk" not in df.columns:
        raise ValueError("CSV must contain 'fall_risk' column. Run infer.py first.")

    df = df.dropna(subset=["fall_risk"])
    y_true = df["fall_label"].values
    y_score = df["fall_risk"].values
    y_pred = (y_score > threshold).astype(int)

    print(f"Samples: {len(y_true)}")
    print(f"Fall rate: {y_true.mean():.3f}")
    print(f"AUC: {roc_auc_score(y_true, y_score):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args.input, args.threshold)


if __name__ == "__main__":
    main()
