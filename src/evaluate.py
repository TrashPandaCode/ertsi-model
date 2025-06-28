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
import argparse

from seed import set_seeds


def evaluate(
    model_path="output/exV3-reverbcnn_synth_only.pt", 
    data_dir="data/test/real",  
    batch_size=32,
    test_data_type="synth"  
):
    set_seeds(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    freqs = [250, 500, 1000, 2000, 4000, 8000]

    if test_data_type == "synth":
        data_dir = "data/test/synth"
    elif test_data_type == "real":
        data_dir = "data/test/real"
    
    print(f"Evaluating on: {data_dir}")
    
    dataset = ReverbRoomDataset(data_dir, freqs=freqs, augment=False)
    
    if len(dataset) == 0:
        print(f"Warnung: Keine Daten in {data_dir} gefunden!")
        return None
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    model = ReverbCNN(num_frequencies=len(freqs))
    
    if not os.path.exists(model_path):
        print(f"Fehler: Modell nicht gefunden unter {model_path}")
        return None
        
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

    # Evaluation Ordner mit Test-Typ benennen
    eval_dir = f"evaluation_{test_data_type}"
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(f"{eval_dir}/plots", exist_ok=True)

    with open(f"{eval_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Rest der Visualisierungen (gleich wie vorher, nur mit eval_dir)
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
    plt.savefig(f"{eval_dir}/plots/predictions_vs_truth.png")

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
    plt.savefig(f"{eval_dir}/plots/error_distribution.png")

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
    plt.savefig(f"{eval_dir}/plots/frequency_examples.png")

    # MC Dropout und weitere Plots (vereinfacht für Übersichtlichkeit)
    print("\nPerforming MC Dropout inference for uncertainty estimation...")
    model.enable_dropout()
    num_mc_samples = 10  # Reduziert für schnellere Ausführung

    mc_preds = []
    with torch.no_grad():
        for _ in range(num_mc_samples):
            preds_batch = []
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds_batch.append(outputs.cpu().numpy())
            mc_preds.append(np.concatenate(preds_batch, axis=0))

    mc_preds = np.stack(mc_preds, axis=0)
    mc_means = mc_preds.mean(axis=0)
    mc_stds = mc_preds.std(axis=0)

    # Uncertainty Distribution
    plt.figure(figsize=(15, 10))
    for i, freq in enumerate(freqs):
        plt.subplot(2, 3, i + 1)
        plt.hist(mc_stds[:, i], bins=30, alpha=0.7)
        plt.title(f"Uncertainty (STD) at {freq} Hz")
        plt.xlabel("Prediction Std Dev (seconds)")
        plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{eval_dir}/plots/uncertainty_distribution.png")

    print(f"\nEvaluation complete. Results saved to '{eval_dir}/' directory.")
    return metrics


if __name__ == "__main__":
    # Argument Parser für einfache Änderung des Test-Typs
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-type", choices=["real", "synth"], default="real", 
                       help="Typ der Testdaten (real oder synth)")
    parser.add_argument("--model", default="output/exV3-reverbcnn_synth_only.pt", 
                       help="Pfad zum Modell")
    args = parser.parse_args()
    
    evaluate(model_path=args.model, test_data_type=args.test_type)