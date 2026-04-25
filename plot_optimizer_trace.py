#!/usr/bin/env python3
"""
Plot train/test loss and accuracy over continual-learning time from loss trace CSVs.

Expected input files are the CSVs written by experiment/accuracy/acc_avg.py:
  *_loss_trace_index*.csv
"""

import argparse
import glob
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = [
    "optimizer",
    "learning_rate",
    "sgd_momentum",
    "batch_size",
    "seed",
    "order_index",
    "task_index",
    "epoch_index",
    "train_loss",
    "train_accuracy",
    "seen_test_loss",
    "seen_test_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot continual train/test metrics for one optimizer from loss trace CSVs."
    )
    parser.add_argument(
        "--results-dir",
        default=".",
        help="Directory containing *_loss_trace_index*.csv files.",
    )
    parser.add_argument(
        "--optimizer",
        required=True,
        choices=["adam", "sgd", "sgd_momentum", "adamw", "rmsprop"],
        help="Optimizer to plot.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional filter for one learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional filter for one batch size.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="Optional filter for one SGD momentum value.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to write plot images.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively.",
    )
    return parser.parse_args()


def load_loss_trace_files(results_dir: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, "*_loss_trace_index*.csv")
    files: List[str] = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No loss trace CSV files found with pattern: {pattern}"
        )

    frames = []
    for path in files:
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        df["source_file"] = os.path.basename(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_global_step(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["epoch_index"] = out["epoch_index"].astype(int)
    out["task_index"] = out["task_index"].astype(int)
    out["seed"] = out["seed"].astype(int)
    out["order_index"] = out["order_index"].astype(int)

    epochs_per_task = int(out["epoch_index"].max()) + 1
    out["global_step"] = out["task_index"] * epochs_per_task + out["epoch_index"]
    return out


def plot_one_setting(df: pd.DataFrame, output_dir: str, show: bool) -> str:
    # Aggregate across seeds/orders/files to give one clean trend for this setting.
    agg = (
        df.groupby("global_step", as_index=False)[
            ["train_loss", "seen_test_loss", "train_accuracy", "seen_test_accuracy"]
        ]
        .mean()
        .sort_values("global_step")
    )

    lr = float(df["learning_rate"].iloc[0])
    mom = float(df["sgd_momentum"].iloc[0])
    bs = int(df["batch_size"].iloc[0])
    opt = str(df["optimizer"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(agg["global_step"], agg["train_loss"], label="train_loss")
    axes[0].plot(agg["global_step"], agg["seen_test_loss"], label="seen_test_loss")
    axes[0].set_title("Loss Through Time")
    axes[0].set_xlabel("Global Step (task*epoch + epoch)")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(agg["global_step"], agg["train_accuracy"], label="train_accuracy")
    axes[1].plot(agg["global_step"], agg["seen_test_accuracy"], label="seen_test_accuracy")
    axes[1].set_title("Accuracy Through Time")
    axes[1].set_xlabel("Global Step (task*epoch + epoch)")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"optimizer={opt} | lr={lr} | momentum={mom} | batch_size={bs}")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_name = f"{opt}_lr{lr}_mom{mom}_bs{bs}_train_test_trace.png".replace(".", "p")
    out_path = os.path.join(output_dir, out_name)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    df = load_loss_trace_files(args.results_dir)

    df = df[df["optimizer"] == args.optimizer]
    if args.learning_rate is not None:
        df = df[df["learning_rate"] == args.learning_rate]
    if args.batch_size is not None:
        df = df[df["batch_size"] == args.batch_size]
    if args.momentum is not None:
        df = df[df["sgd_momentum"] == args.momentum]

    if df.empty:
        raise ValueError("No rows left after filtering. Adjust optimizer/filter arguments.")

    df = compute_global_step(df)

    # Plot one figure per hyperparameter setting remaining.
    setting_cols = ["optimizer", "learning_rate", "sgd_momentum", "batch_size"]
    outputs = []
    for _, subset in df.groupby(setting_cols, sort=True):
        outputs.append(plot_one_setting(subset, args.output_dir, args.show))

    print("Saved plot(s):")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()

