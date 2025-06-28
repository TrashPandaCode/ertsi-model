from dataset import ReverbRoomDataset
from model import ReverbCNN
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import pandas as pd

from seed import set_seeds

experiment = "exp7"


def evaluate(
    model_path=f"output/{experiment}-reverbcnn.pt",
    data_dir="data/test/real",
    batch_size=32,
):
    set_seeds(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    freqs = [250, 500, 1000, 2000, 4000, 8000]

    dataset = ReverbRoomDataset(data_dir, freqs=freqs, augment=False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    model = ReverbCNN(num_frequencies=len(freqs))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Initialize variables to store predictions and targets
    all_preds = []
    all_targets = []

    # Evaluate model
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            # Store predictions and targets
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics per frequency
    metrics = {"frequency": {}, "overall": {}}

    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    metrics["overall"] = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

    # Calculate metrics for each frequency
    for i, freq in enumerate(freqs):
        freq_preds = all_preds[:, i]
        freq_targets = all_targets[:, i]

        freq_mse = mean_squared_error(freq_targets, freq_preds)
        freq_rmse = np.sqrt(freq_mse)
        freq_mae = mean_absolute_error(freq_targets, freq_preds)
        freq_r2 = r2_score(freq_targets, freq_preds)

        metrics["frequency"][str(freq)] = {
            "mse": float(freq_mse),
            "rmse": float(freq_rmse),
            "mae": float(freq_mae),
            "r2": float(freq_r2),
        }

    print("\n=== Overall Performance ===")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    print("\n=== Performance by Frequency ===")
    for freq in freqs:
        freq_metrics = metrics["frequency"][str(freq)]
        print(f"\nFrequency: {freq} Hz")
        print(f"MSE: {freq_metrics['mse']:.4f}")
        print(f"RMSE: {freq_metrics['rmse']:.4f}")
        print(f"MAE: {freq_metrics['mae']:.4f}")
        print(f"R²: {freq_metrics['r2']:.4f}")

    os.makedirs(f"evaluation_{experiment}", exist_ok=True)

    with open(f"evaluation_{experiment}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    os.makedirs(f"evaluation_{experiment}/plots", exist_ok=True)

    # Plot predictions vs ground truth for each frequency
    plt.figure(figsize=(15, 10))
    for i, freq in enumerate(freqs):
        plt.subplot(2, 3, i + 1)
        plt.scatter(all_targets[:, i], all_preds[:, i], alpha=0.3)

        # Plot perfect prediction line
        min_val = min(all_targets[:, i].min(), all_preds[:, i].min())
        max_val = max(all_targets[:, i].max(), all_preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")

        plt.title(f"Frequency: {freq} Hz")
        plt.xlabel("Ground Truth RT60 (s)")
        plt.ylabel("Predicted RT60 (s)")

        # Add metrics to plot
        freq_metrics = metrics["frequency"][str(freq)]
        plt.text(
            0.05,
            0.95,
            f"MSE: {freq_metrics['mse']:.4f}\nRMSE: {freq_metrics['rmse']:.4f}\nMAE: {freq_metrics['mae']:.4f}\nR²: {freq_metrics['r2']:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(f"evaluation_{experiment}/plots/predictions_vs_truth.png")

    # Plot error distribution for each frequency
    plt.figure(figsize=(15, 10))
    for i, freq in enumerate(freqs):
        errors = all_preds[:, i] - all_targets[:, i]

        plt.subplot(2, 3, i + 1)
        plt.hist(errors, bins=30, alpha=0.7)
        plt.title(f"Error Distribution: {freq} Hz")
        plt.xlabel("Prediction Error (seconds)")
        plt.ylabel("Count")

        # Add mean and std to plot
        mean_error = errors.mean()
        std_error = errors.std()
        plt.axvline(
            mean_error, color="r", linestyle="--", label=f"Mean: {mean_error:.3f}s"
        )
        plt.axvline(
            mean_error + std_error,
            color="g",
            linestyle=":",
            label=f"Std: {std_error:.3f}s",
        )
        plt.axvline(mean_error - std_error, color="g", linestyle=":")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"evaluation_{experiment}/plots/error_distribution.png")

    # Plot RT60 by frequency for a few random examples
    num_examples = min(5, len(all_targets))
    indices = np.random.choice(len(all_targets), num_examples, replace=False)

    plt.figure(figsize=(12, 8))
    for idx in indices:
        plt.plot(freqs, all_targets[idx], "o-", label=f"Truth {idx}")
        plt.plot(freqs, all_preds[idx], "x--", label=f"Pred {idx}")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("RT60 (s)")
    plt.title("RT60 by Frequency: Ground Truth vs Predictions")
    plt.xscale("log")
    plt.xticks(freqs, [str(f) for f in freqs])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"evaluation_{experiment}/plots/frequency_examples.png")

    # ---------------------------
    # Continuous Error Heatmap
    # ---------------------------
    print("\nGenerating Continuous Error Heatmap...")
    plt.figure(figsize=(15, 10))
    for i, freq in enumerate(freqs):
        plt.subplot(2, 3, i + 1)

        df = pd.DataFrame(
            {"Ground Truth": all_targets[:, i], "Prediction": all_preds[:, i]}
        )

        sns.histplot(
            data=df,
            x="Ground Truth",
            y="Prediction",
            bins=30,
            pthresh=0.01,
            cmap="viridis",
        )

        plt.plot(
            [df["Ground Truth"].min(), df["Ground Truth"].max()],
            [df["Ground Truth"].min(), df["Ground Truth"].max()],
            "r--",
            label="Ideal",
        )
        plt.xlabel("Ground Truth RT60 (s)")
        plt.ylabel("Predicted RT60 (s)")
        plt.title(f"RT60 Density Heatmap @ {freq} Hz")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"evaluation_{experiment}/plots/rt60_heatmap_density.png")

    plt.figure(figsize=(15, 10))
    for i, freq in enumerate(freqs):
        plt.subplot(2, 3, i + 1)

        df = pd.DataFrame(
            {"Ground Truth": all_targets[:, i], "Prediction": all_preds[:, i]}
        )

        # Joint KDE plot with contours
        sns.kdeplot(
            data=df,
            x="Ground Truth",
            y="Prediction",
            fill=True,
            cmap="magma",
            thresh=0.05,
            levels=100,
        )

        # Ideal line (perfect prediction)
        min_val = min(df["Ground Truth"].min(), df["Prediction"].min())
        max_val = max(df["Ground Truth"].max(), df["Prediction"].max())
        plt.plot([min_val, max_val], [min_val, max_val], "c--", label="Ideal")

        plt.xlabel("Ground Truth RT60 (s)")
        plt.ylabel("Predicted RT60 (s)")
        plt.title(f"KDE Heatmap @ {freq} Hz")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"evaluation_{experiment}/plots/rt60_kde_heatmap.png")

    print(f"\nEvaluation complete. Results saved to 'evaluation/' directory.")
    return metrics


if __name__ == "__main__":
    evaluate()
