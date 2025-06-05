import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
from pathlib import Path

from dataset import ReverbRoomDataset
from model import ReverbCNN
from seed import set_seeds


def evaluate(
    model_path="output/reverbcnn_simultaneous.pt",
    data_dir="data/test/real",
    batch_size=32,
    num_mc_samples=20,
    model_type="simultaneous"
):
    """
    Evaluate the improved ReverbCNN model
    
    Args:
        model_path: Path to the trained model
        data_dir: Directory containing test data
        batch_size: Batch size for evaluation
        num_mc_samples: Number of Monte Carlo dropout samples
        model_type: Type of model ('simultaneous', 'progressive', 'balanced')
    """
    
    set_seeds(42)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freqs = [250, 500, 1000, 2000, 4000, 8000]
    
    print(f"=== Evaluating {model_type.title()} Model ===")
    print(f"Model path: {model_path}")
    print(f"Data directory: {data_dir}")
    print(f"Device: {device}")
    print(f"MC samples for uncertainty: {num_mc_samples}")
    
    # Create dataset
    dataset = ReverbRoomDataset(
        data_dir, 
        freqs=freqs, 
        augment=False,
    )
    
    if len(dataset) == 0:
        print(f"❌ No test data found in {data_dir}")
        return None
    
    print(f"✅ Test dataset loaded: {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Load model with improved architecture
    model = ReverbCNN(
        num_frequencies=len(freqs),
        learning_rate=0.001,  # Not used for evaluation
        frequencies=freqs,
        dropout_rate=0.2,
        use_scheduler=False
    )
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    model.to(device)
    model.eval()
    
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
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(all_preds, all_targets, freqs, model_type)
    
    # Monte Carlo Dropout for uncertainty estimation
    print(f"\n=== Monte Carlo Dropout Evaluation (n={num_mc_samples}) ===")
    mc_means, mc_stds = perform_mc_dropout_evaluation(
        model, dataloader, device, num_mc_samples
    )
    
    # Create visualizations
    print("\n=== Generating Visualizations ===")
    create_evaluation_plots(
        all_preds, all_targets, mc_means, mc_stds, freqs, model_type
    )
    
    # Save comprehensive results
    save_evaluation_results(metrics, model_path, model_type, len(all_preds))
    
    # Print summary
    print_evaluation_summary(metrics, model_type)
    
    return metrics


def calculate_comprehensive_metrics(all_preds, all_targets, freqs, model_type):
    """Calculate comprehensive evaluation metrics"""
    
    metrics = {
        "frequency": {},
        "overall": {},
        "correlations": {},
        "model_info": {
            "type": model_type,
            "num_samples": len(all_preds),
            "frequencies": freqs
        }
    }
    
    # Overall metrics
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    # Additional overall metrics
    mape = np.mean(np.abs((all_targets - all_preds) / all_targets)) * 100  # Mean Absolute Percentage Error
    max_error = np.max(np.abs(all_targets - all_preds))
    median_error = np.median(np.abs(all_targets - all_preds))
    
    metrics["overall"] = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
        "max_error": float(max_error),
        "median_error": float(median_error)
    }
    
    # Per-frequency metrics
    for i, freq in enumerate(freqs):
        freq_preds = all_preds[:, i]
        freq_targets = all_targets[:, i]
        
        freq_mse = mean_squared_error(freq_targets, freq_preds)
        freq_rmse = np.sqrt(freq_mse)
        freq_mae = mean_absolute_error(freq_targets, freq_preds)
        freq_r2 = r2_score(freq_targets, freq_preds)
        freq_mape = np.mean(np.abs((freq_targets - freq_preds) / freq_targets)) * 100
        
        # Correlation coefficient
        freq_corr = np.corrcoef(freq_targets, freq_preds)[0, 1]
        
        metrics["frequency"][str(freq)] = {
            "mse": float(freq_mse),
            "rmse": float(freq_rmse),
            "mae": float(freq_mae),
            "r2": float(freq_r2),
            "mape": float(freq_mape),
            "correlation": float(freq_corr)
        }
    
    # Cross-frequency correlations
    pred_corr_matrix = np.corrcoef(all_preds.T)
    target_corr_matrix = np.corrcoef(all_targets.T)
    corr_diff = np.linalg.norm(pred_corr_matrix - target_corr_matrix, 'fro')
    
    metrics["correlations"] = {
        "correlation_matrix_difference": float(corr_diff),
        "predicted_correlations": pred_corr_matrix.tolist(),
        "target_correlations": target_corr_matrix.tolist()
    }
    
    return metrics


def perform_mc_dropout_evaluation(model, dataloader, device, num_mc_samples):
    """Perform Monte Carlo dropout evaluation for uncertainty estimation"""
    
    model.train()  # Enable dropout
    model.enable_dropout()  # Custom method to enable dropout during eval
    
    mc_preds = []
    
    for mc_iter in range(num_mc_samples):
        iter_preds = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                iter_preds.append(outputs.cpu().numpy())
        
        mc_preds.append(np.concatenate(iter_preds, axis=0))
        if (mc_iter + 1) % 5 == 0:
            print(f"MC iteration {mc_iter + 1}/{num_mc_samples} completed")
    
    mc_preds = np.array(mc_preds)  # Shape: [mc_samples, num_examples, num_freqs]
    mc_means = mc_preds.mean(axis=0)
    mc_stds = mc_preds.std(axis=0)
    
    print(f"Uncertainty analysis completed")
    print(f"Mean uncertainty (std): {mc_stds.mean():.4f}")
    print(f"Max uncertainty: {mc_stds.max():.4f}")
    
    return mc_means, mc_stds


def create_evaluation_plots(all_preds, all_targets, mc_means, mc_stds, freqs, model_type):
    """Create comprehensive evaluation plots"""
    
    # Ensure output directory exists
    output_dir = Path("evaluation/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Predictions vs Ground Truth with improved layout
    plt.figure(figsize=(20, 14))
    for i, freq in enumerate(freqs):
        plt.subplot(2, 3, i + 1)
        
        # Main scatter plot
        plt.scatter(all_targets[:, i], all_preds[:, i], alpha=0.6, s=25, c='blue', edgecolors='navy', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(all_targets[:, i].min(), all_preds[:, i].min())
        max_val = max(all_targets[:, i].max(), all_preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=2, label='Perfect')
        
        # Calculate metrics for this frequency
        freq_rmse = np.sqrt(mean_squared_error(all_targets[:, i], all_preds[:, i]))
        freq_mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        freq_r2 = r2_score(all_targets[:, i], all_preds[:, i])
        freq_corr = np.corrcoef(all_targets[:, i], all_preds[:, i])[0, 1]
        
        plt.title(f"{freq} Hz - {model_type.title()} Model", fontsize=14, fontweight='bold')
        plt.xlabel("Ground Truth RT60 (s)", fontsize=12)
        plt.ylabel("Predicted RT60 (s)", fontsize=12)
        
        # Enhanced metrics display
        metrics_text = (f"RMSE: {freq_rmse:.3f}\n"
                       f"MAE: {freq_mae:.3f}\n" 
                       f"R²: {freq_r2:.3f}\n"
                       f"ρ: {freq_corr:.3f}")
        
        plt.text(0.05, 0.95, metrics_text,
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
                fontsize=10, fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.suptitle(f"{model_type.title()} Model Evaluation", 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_predictions_vs_truth.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Enhanced Error Distribution
    plt.figure(figsize=(20, 14))
    for i, freq in enumerate(freqs):
        errors = all_preds[:, i] - all_targets[:, i]
        
        plt.subplot(2, 3, i + 1)
        
        # Histogram with KDE overlay
        n, bins, patches = plt.hist(errors, bins=40, alpha=0.7, color='skyblue', 
                                   edgecolor='black', density=True)
        
        # KDE overlay
        try:
            from scipy import stats
            kde = stats.gaussian_kde(errors)
            x_range = np.linspace(errors.min(), errors.max(), 200)
            plt.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except ImportError:
            pass
        
        plt.title(f"Error Distribution: {freq} Hz", fontsize=14, fontweight='bold')
        plt.xlabel("Prediction Error (seconds)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        
        # Statistics
        mean_error = errors.mean()
        std_error = errors.std()
        median_error = np.median(errors)
        q25, q75 = np.percentile(errors, [25, 75])
        
        plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_error:.3f}s')
        plt.axvline(median_error, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_error:.3f}s')
        plt.axvline(q25, color='orange', linestyle=':', alpha=0.7, label=f'Q25: {q25:.3f}s')
        plt.axvline(q75, color='orange', linestyle=':', alpha=0.7, label=f'Q75: {q75:.3f}s')
        
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
    
    plt.suptitle(f"{model_type.title()} Model Error Analysis", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_error_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Advanced Uncertainty Visualization
    plt.figure(figsize=(20, 14))
    for i, freq in enumerate(freqs):
        plt.subplot(2, 3, i + 1)
        
        # Scatter plot colored by uncertainty
        scatter = plt.scatter(all_targets[:, i], mc_means[:, i], c=mc_stds[:, i], 
                            alpha=0.7, s=30, cmap='viridis', edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='Prediction Uncertainty (std)')
        
        # Perfect prediction line
        min_val = min(all_targets[:, i].min(), mc_means[:, i].min())
        max_val = max(all_targets[:, i].max(), mc_means[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=2)
        
        plt.title(f"Uncertainty Analysis: {freq} Hz", fontsize=14, fontweight='bold')
        plt.xlabel("Ground Truth RT60 (s)", fontsize=12)
        plt.ylabel("Mean Prediction (s)", fontsize=12)
        
        # Uncertainty statistics
        mean_uncertainty = mc_stds[:, i].mean()
        max_uncertainty = mc_stds[:, i].max()
        
        plt.text(0.05, 0.95, f"Mean σ: {mean_uncertainty:.3f}\nMax σ: {max_uncertainty:.3f}",
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                fontsize=10)
        
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f"{model_type.title()} Model Uncertainty Analysis (MC Dropout)", 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_uncertainty_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. RT60 Frequency Response Examples
    num_examples = min(8, len(all_targets))
    indices = np.random.choice(len(all_targets), num_examples, replace=False)
    
    plt.figure(figsize=(16, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, num_examples))
    
    for idx, color in zip(indices, colors):
        plt.errorbar(freqs, mc_means[idx], yerr=mc_stds[idx], 
                    fmt='x--', color=color, alpha=0.8, capsize=5, capthick=2,
                    label=f'Pred {idx}', linewidth=2)
        plt.plot(freqs, all_targets[idx], 'o-', color=color, alpha=0.9, 
                label=f'Truth {idx}', linewidth=2, markersize=8)
    
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("RT60 (s)", fontsize=14)
    plt.title(f"{model_type.title()} Model: RT60 Frequency Response (Sample Cases)", 
              fontsize=16, fontweight='bold')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_rt60_frequency_response.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Advanced Density Heatmaps
    plt.figure(figsize=(20, 14))
    for i, freq in enumerate(freqs):
        plt.subplot(2, 3, i + 1)
        
        # Create 2D histogram with better binning
        counts, xbins, ybins = np.histogram2d(all_targets[:, i], all_preds[:, i], bins=30)
        
        # Plot heatmap
        plt.imshow(counts.T, origin='lower', cmap='viridis', aspect='auto',
                  extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])
        plt.colorbar(label='Count')
        
        # Perfect prediction line
        min_val = min(all_targets[:, i].min(), all_preds[:, i].min())
        max_val = max(all_targets[:, i].max(), all_preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=3)
        
        plt.xlabel("Ground Truth RT60 (s)", fontsize=12)
        plt.ylabel("Predicted RT60 (s)", fontsize=12)
        plt.title(f"Density Heatmap: {freq} Hz", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f"{model_type.title()} Model Prediction Density Analysis", 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_type}_density_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ All visualizations saved successfully")


def save_evaluation_results(metrics, model_path, model_type, num_samples):
    """Save comprehensive evaluation results"""
    
    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)
    
    # Add additional metadata
    metrics["model_info"].update({
        "model_path": str(model_path),
        "model_type": model_type,
        "evaluation_samples": num_samples
    })
    
    # Save metrics
    with open(output_dir / f"{model_type}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save summary table
    summary_data = []
    for freq in metrics["frequency"].keys():
        freq_metrics = metrics["frequency"][freq]
        summary_data.append({
            "Frequency (Hz)": freq,
            "RMSE": freq_metrics["rmse"],
            "MAE": freq_metrics["mae"],
            "R²": freq_metrics["r2"],
            "MAPE (%)": freq_metrics["mape"],
            "Correlation": freq_metrics["correlation"]
        })
    
    # Add overall metrics
    overall = metrics["overall"]
    summary_data.append({
        "Frequency (Hz)": "Overall",
        "RMSE": overall["rmse"],
        "MAE": overall["mae"],
        "R²": overall["r2"],
        "MAPE (%)": overall["mape"],
        "Correlation": "N/A"
    })
    
    # Save as CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / f"{model_type}_summary.csv", index=False)
    
    print(f"✅ Results saved to evaluation/{model_type}_*.json/csv")


def print_evaluation_summary(metrics, model_type):
    """Print comprehensive evaluation summary"""
    
    overall = metrics["overall"]
    
    print(f"\n{'='*60}")
    print(f"{model_type.title()} Model Evaluation Summary")
    print(f"{'='*60}")
    
    print(f"\n📊 Overall Performance:")
    print(f"   RMSE: {overall['rmse']:.4f} seconds")
    print(f"   MAE:  {overall['mae']:.4f} seconds")
    print(f"   R²:   {overall['r2']:.4f}")
    print(f"   MAPE: {overall['mape']:.2f}%")
    print(f"   Max Error: {overall['max_error']:.4f} seconds")
    print(f"   Median Error: {overall['median_error']:.4f} seconds")
    
    print(f"\n🎵 Performance by Frequency:")
    freq_header = f"{'Freq (Hz)':<10} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'MAPE (%)':<10} {'Corr':<8}"
    print(freq_header)
    print("-" * len(freq_header))
    
    for freq, freq_metrics in metrics["frequency"].items():
        print(f"{freq:<10} {freq_metrics['rmse']:<8.4f} {freq_metrics['mae']:<8.4f} "
              f"{freq_metrics['r2']:<8.4f} {freq_metrics['mape']:<10.2f} "
              f"{freq_metrics['correlation']:<8.4f}")
    
    print(f"\n🔗 Cross-frequency Correlation:")
    print(f"   Matrix Difference (Frobenius): {metrics['correlations']['correlation_matrix_difference']:.4f}")
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"   Results saved to: evaluation/{model_type}_*")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Improved ReverbCNN Model")
    parser.add_argument("--model_path", default="output/reverbcnn_simultaneous.pt",
                       help="Path to the trained model")
    parser.add_argument("--data_dir", default="data/test/real",
                       help="Test data directory")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--mc_samples", type=int, default=20,
                       help="Number of MC dropout samples")
    parser.add_argument("--model_type", default="simultaneous",
                       choices=["simultaneous", "progressive", "balanced"],
                       help="Type of model being evaluated")
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_mc_samples=args.mc_samples,
        model_type=args.model_type
    )