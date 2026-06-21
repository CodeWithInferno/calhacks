"""Plot training log CSV."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot(log_path, output_path=None):
    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0, 0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend()
    axes[0, 0].set_title("Loss")

    axes[0, 1].plot(df["epoch"], df["val_auc"], label="AUC")
    axes[0, 1].plot(df["epoch"], df["val_ap"], label="AP")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("score")
    axes[0, 1].legend()
    axes[0, 1].set_title("Validation Ranking Metrics")

    axes[1, 0].plot(df["epoch"], df["val_acc"], label="accuracy")
    axes[1, 0].plot(df["epoch"], df["val_f1"], label="F1")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("score")
    axes[1, 0].legend()
    axes[1, 0].set_title("Validation Classification Metrics")

    axes[1, 1].plot(df["epoch"], df["lr"])
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("learning rate")
    axes[1, 1].set_title("LR")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="models/world_model/training_log.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    plot(args.log, args.output)


if __name__ == "__main__":
    main()
