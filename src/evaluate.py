import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os

from dataset import ReverbRoomDataset
from model import ReverbCNN
from seed import set_seeds


def evaluate_progressive(
    model_path="output/reverbcnn_progressive.pt",
    data_dir="data/test/real", 
    batch_size=16,
    resolution=448,
    num_mc_samples=10
):
    """
    Evaluate the progressive resolution trained model
    
    Args:
        model_path: Path to the progressive model weights
        data_dir: Directory containing test data
        batch_size: Batch size for evaluation
        resolution: Image resolution to use (should match final training resolution)
        num_mc_samples: Number of Monte Carlo dropout samples for uncertainty
    """
    
    set_seeds(42)
    
    # Ensure output directory exists
    os.makedirs("evaluation/plots", exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freqs = [250, 500, 1000, 2000, 4000, 8000]
    
    print(f"Loading progressive model from: {model_path}")
    print(f"Using resolution: {resolution}x{resolution}")
    print(f"Device: {device}")
    
    model = ReverbCNN(num_frequencies=len(freqs))
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Create dataset with the same resolution used in final training stage
    print(f"Loading test dataset from: {data_dir}")
    dataset = ReverbRoomDataset(
        data_dir, 
        freqs=freqs, 
        augment=False,
        image_size=resolution
    )
    
    if len(dataset) == 0:
        print(f"❌ No test data found in {data_dir}")
        return
    
    print(f"✅ Test dataset loaded: {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Standard evaluation
    print("\n=== Standard Evaluation ===")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"Evaluation completed: {len(all_preds)} samples processed")
    
    # Calculate metrics per frequency
    metrics = {"frequency": {}, "overall": {}}
    
    # Overall metrics
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
    
    # Per-frequency metrics
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
    
    # Print results
    print("\n=== Progressive Model Performance ===")
    print(f"Overall MSE: {mse:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Overall MAE: {mae:.4f}")
    print(f"Overall R²: {r2:.4f}")
    
    print("\n=== Performance by Frequency ===")
    for freq in freqs:
        freq_metrics = metrics["frequency"][str(freq)]
        print(f"{freq} Hz - RMSE: {freq_metrics['rmse']:.4f}, MAE: {freq_metrics['mae']:.4f}, R²: {freq_metrics['r2']:.4f}")
    
    # Monte Carlo Dropout Evaluation for Uncertainty
    print(f"\n=== Monte Carlo Dropout Evaluation (n={num_mc_samples}) ===")
    model.train()  # Enable dropout
    
    mc_preds = []
    for mc_iter in range(num_mc_samples):
        iter_preds = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                iter_preds.append(outputs.cpu().numpy())
        
        mc_preds.append(np.concatenate(iter_preds, axis=0))
        print(f"MC iteration {mc_iter + 1}/{num_mc_samples} completed")
    
    mc_preds = np.array(mc_preds)  # Shape: [mc_samples, num_examples, num_freqs]
    mc_means = mc_preds.mean(axis=0)
    mc_stds = mc_preds.std(axis=0)
    
    # Visualization
    print("\n=== Generating Visualizations ===")
    
    # 1. Predictions vs Ground Truth
    plt.figure(figsize=(18, 12))
    for i, freq in enumerate(freqs):
        plt.subplot(2, 3, i + 1)
        plt.scatter(all_targets[:, i], all_preds[:, i], alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(all_targets[:, i].min(), all_preds[:, i].min())
        max_val = max(all_targets[:, i].max(), all_preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8)
        
        plt.title(f"Progressive Model: {freq} Hz")
        plt.xlabel("Ground Truth RT60 (s)")
        plt.ylabel("Predicted RT60 (s)")
        
        # Add metrics
        freq_metrics = metrics["frequency"][str(freq)]
        plt.text(0.05, 0.95, 
                f"RMSE: {freq_metrics['rmse']:.3f}\nMAE: {freq_metrics['mae']:.3f}\nR²: {freq_metrics['r2']:.3f}",
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.suptitle(f"Progressive Model Evaluation (Resolution: {resolution}x{resolution})", fontsize=16)
    plt.tight_layout()
    plt.savefig("evaluation/plots/progressive_predictions_vs_truth.png", dpi=150)
    print("✅ Saved: progressive_predictions_vs_truth.png")
    
    # 2. Error Distribution
    plt.figure(figsize=(18, 12))
    for i, freq in enumerate(freqs):
        errors = all_preds[:, i] - all_targets[:, i]
        
        plt.subplot(2, 3, i + 1)
        plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f"Error Distribution: {freq} Hz")
        plt.xlabel("Prediction Error (seconds)")
        plt.ylabel("Count")
        
        # Add statistics
        mean_error = errors.mean()
        std_error = errors.std()
        plt.axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.3f}s')
        plt.axvline(mean_error + std_error, color='green', linestyle=':', label=f'±1σ: {std_error:.3f}s')
        plt.axvline(mean_error - std_error, color='green', linestyle=':')
        plt.legend()
    
    plt.suptitle(f"Progressive Model Error Analysis (Resolution: {resolution}x{resolution})", fontsize=16)
    plt.tight_layout()
    plt.savefig("evaluation/plots/progressive_error_distribution.png", dpi=150)
    print("✅ Saved: progressive_error_distribution.png")
    
    # 3. Uncertainty Visualization
    plt.figure(figsize=(18, 12))
    for i, freq in enumerate(freqs):
        plt.subplot(2, 3, i + 1)
        scatter = plt.scatter(all_targets[:, i], mc_means[:, i], c=mc_stds[:, i], 
                   alpha=0.6, s=20, cmap='viridis')
        plt.colorbar(scatter, label='Prediction Uncertainty (std)')
        
        # Perfect prediction line
        min_val = min(all_targets[:, i].min(), mc_means[:, i].min())
        max_val = max(all_targets[:, i].max(), mc_means[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8)
        
        plt.title(f"Uncertainty: {freq} Hz")
        plt.xlabel("Ground Truth RT60 (s)")
        plt.ylabel("Mean Prediction (s)")
    
    plt.suptitle(f"Progressive Model Uncertainty Analysis (MC Dropout)", fontsize=16)
    plt.tight_layout()
    plt.savefig("evaluation/plots/progressive_uncertainty_analysis.png", dpi=150)
    print("✅ Saved: progressive_uncertainty_analysis.png")
    
    # 4. RT60 by Frequency for Sample Cases
    num_examples = min(8, len(all_targets))
    indices = np.random.choice(len(all_targets), num_examples, replace=False)
    
    plt.figure(figsize=(15, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, num_examples))
    
    for idx, color in zip(indices, colors):
        plt.plot(freqs, all_targets[idx], 'o-', color=color, alpha=0.7, 
                label=f'Truth {idx}', linewidth=2)
        plt.plot(freqs, all_preds[idx], 'x--', color=color, alpha=0.7, 
                label=f'Pred {idx}', linewidth=2)
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("RT60 (s)")
    plt.title(f"Progressive Model: RT60 by Frequency (Sample Cases)")
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("evaluation/plots/progressive_rt60_by_frequency.png", dpi=150, bbox_inches='tight')
    print("✅ Saved: progressive_rt60_by_frequency.png")
    
    # 5. Density Heatmap
    plt.figure(figsize=(18, 12))
    for i, freq in enumerate(freqs):
        plt.subplot(2, 3, i + 1)
        
        df = pd.DataFrame({
            "Ground Truth": all_targets[:, i], 
            "Prediction": all_preds[:, i]
        })
        
        sns.histplot(
            data=df,
            x="Ground Truth",
            y="Prediction",
            bins=30,
            pthresh=0.01,
            cmap="viridis"
        )
        
        plt.plot(
            [df["Ground Truth"].min(), df["Ground Truth"].max()],
            [df["Ground Truth"].min(), df["Ground Truth"].max()],
            "r--",
            label="Ideal"
        )
        plt.xlabel("Ground Truth RT60 (s)")
        plt.ylabel("Predicted RT60 (s)")
        plt.title(f"Density Heatmap: {freq} Hz")
        plt.legend()
    
    plt.suptitle(f"Progressive Model Density Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig("evaluation/plots/progressive_density_heatmap.png", dpi=150)
    print("✅ Saved: progressive_density_heatmap.png")
    
    # Save metrics to JSON
    metrics["model_info"] = {
        "model_path": model_path,
        "resolution": resolution,
        "test_samples": len(all_preds),
        "mc_samples": num_mc_samples,
        "training_type": "progressive_resolution"
    }
    
    with open("evaluation/progressive_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("✅ Saved: progressive_metrics.json")
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Progressive model performance:")
    print(f"- Overall RMSE: {rmse:.4f} seconds")
    print(f"- Overall MAE: {mae:.4f} seconds") 
    print(f"- Overall R²: {r2:.4f}")
    print(f"- Resolution used: {resolution}x{resolution}")
    print(f"- Training approach: Progressive (224px → 320px → 448px)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Progressive Resolution Model")
    parser.add_argument("--model_path", default="output/reverbcnn_progressive.pt", 
                       help="Path to progressive model")
    parser.add_argument("--data_dir", default="data/test/real", 
                       help="Test data directory")
    parser.add_argument("--resolution", type=int, default=448, 
                       help="Image resolution for evaluation")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for evaluation")
    parser.add_argument("--mc_samples", type=int, default=10, 
                       help="Number of MC dropout samples")
    
    args = parser.parse_args()
    
    evaluate_progressive(
        model_path=args.model_path,
        data_dir=args.data_dir,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_mc_samples=args.mc_samples
    )