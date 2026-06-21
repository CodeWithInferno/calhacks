"""Benchmark multiple model configs and summarize results."""

import os
import yaml
import json
import argparse
import time
import importlib.util
import pandas as pd


def load_train_function():
    spec = importlib.util.spec_from_file_location("train_world_model", "src/train_world_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.train


def benchmark(config_dir, configs, output_csv, output_json):
    train = load_train_function()
    results = []

    for name in configs:
        cfg_path = os.path.join(config_dir, f"{name}.yaml")
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"{'='*60}")

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        t0 = time.time()
        train(cfg)
        elapsed = time.time() - t0

        # Read final metrics.
        model_dir = cfg["output_dir"]
        final_ckpt_path = os.path.join(model_dir, "final_checkpoint.pt")
        log_path = os.path.join(model_dir, "training_log.csv")

        log_df = pd.read_csv(log_path)
        best_auc = log_df["val_auc"].max()
        final_auc = log_df["val_auc"].iloc[-1]
        final_f1 = log_df["val_f1"].iloc[-1]
        epochs = int(log_df["epoch"].iloc[-1])

        # Parameter count from config + schema.
        with open("src/schema.yaml") as f:
            schema = yaml.safe_load(f)
        input_dim = len(schema["feature_cols"])

        import torch
        from train_world_model import FallPredictorGRU
        model = FallPredictorGRU(input_dim, cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"])
        num_params = sum(p.numel() for p in model.parameters())

        result = {
            "config": name,
            "params": num_params,
            "hidden_dim": cfg["hidden_dim"],
            "num_layers": cfg["num_layers"],
            "batch_size": cfg["batch_size"],
            "epochs": epochs,
            "best_val_auc": best_auc,
            "final_val_auc": final_auc,
            "final_val_f1": final_f1,
            "train_time_sec": elapsed,
        }
        results.append(result)
        print(json.dumps(result, indent=2))

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark summary saved to {output_csv} and {output_json}")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--configs", nargs="+", default=["small", "medium", "large", "xl"])
    parser.add_argument("--output-csv", default="results/benchmark.csv")
    parser.add_argument("--output-json", default="results/benchmark.json")
    args = parser.parse_args()
    benchmark(args.config_dir, args.configs, args.output_csv, args.output_json)


if __name__ == "__main__":
    main()
