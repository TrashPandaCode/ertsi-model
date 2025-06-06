import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import ReverbCNNComparison
from dataset import ReverbRoomDataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from seed import set_seeds


def load_model_with_compatibility(model, model_path, device):
    """Load model state dict with compatibility for different architectures"""
    try:
        # Try loading normally first
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print(f"  ✅ Loaded model successfully with strict=True")
        return True
    except RuntimeError as e:
        if "Unexpected key(s)" in str(e) or "Missing key(s)" in str(e):
            print(f"  ⚠️  Architecture mismatch detected, attempting compatibility loading...")
            
            # Try loading with strict=False to ignore missing/unexpected keys
            try:
                state_dict = torch.load(model_path, map_location=device)
                
                # Filter out incompatible keys
                model_keys = set(model.state_dict().keys())
                filtered_state_dict = {}
                skipped_keys = []
                
                for key, value in state_dict.items():
                    if key in model_keys:
                        # Check if tensor shapes match
                        expected_shape = model.state_dict()[key].shape
                        if value.shape == expected_shape:
                            filtered_state_dict[key] = value
                        else:
                            print(f"    Shape mismatch for {key}: expected {expected_shape}, got {value.shape}")
                            skipped_keys.append(key)
                    else:
                        skipped_keys.append(key)
                
                if skipped_keys:
                    print(f"    Skipped {len(skipped_keys)} incompatible keys")
                
                # Load the filtered state dict
                missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
                
                if missing_keys:
                    print(f"    Warning: {len(missing_keys)} keys will use random initialization")
                
                print(f"  ✅ Loaded {len(filtered_state_dict)} compatible parameters")
                return True
                
            except Exception as e2:
                print(f"    ❌ Compatibility loading failed: {e2}")
                return False
        else:
            print(f"  ❌ Loading failed: {e}")
            return False


def evaluate_all_backbones():
    """Evaluate all trained backbone models"""

    set_seeds(42)

    backbones = [
        # "resnet50_places365",  # Add the original baseline
        "dinov2",
        "clip",  # OpenCLIP ViT-B-32
        "clip_vit_l",  # OpenCLIP ViT-L-14
        "efficientnet_b4",
        "clip_convnext",
    ]
    freqs = [250, 500, 1000, 2000, 4000, 8000]

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = ReverbRoomDataset("data/test/real", freqs=freqs, augment=False)
    
    # Use smaller batch size to avoid memory issues with different architectures
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Found {len(test_dataset)} test samples")

    results = {}
    successful_evaluations = 0

    for backbone_name in backbones:
        print(f"\n{'='*60}")
        print(f"Evaluating {backbone_name.upper()}")
        print(f"{'='*60}")

        model_path = f"comparison_results/{backbone_name}_final.pt"
        if not os.path.exists(model_path):
            print(f"  ❌ Model not found: {model_path}")
            continue

        try:
            # Load model with error handling
            print(f"  📁 Loading model from {model_path}")
            model = ReverbCNNComparison(
                backbone_name=backbone_name,
                num_frequencies=len(freqs),
                frequencies=freqs,
                dropout_rate=0.15,  # Match training configuration
            )
            
            # Use compatibility loading
            if not load_model_with_compatibility(model, model_path, device):
                print(f"  ❌ Failed to load model for {backbone_name}")
                continue
                
            model.to(device)
            model.eval()

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Evaluate with error handling
            all_preds = []
            all_targets = []
            successful_batches = 0
            
            print(f"  🔄 Running inference on {len(test_loader)} batches...")
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    try:
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        # Debug: Print input shape for first batch
                        if batch_idx == 0:
                            print(f"    Input shape: {inputs.shape}")
                            print(f"    Target shape: {targets.shape}")
                        
                        outputs = model(inputs)
                        
                        # Debug: Print output shape for first batch
                        if batch_idx == 0:
                            print(f"    Output shape: {outputs.shape}")
                        
                        all_preds.append(outputs.cpu().numpy())
                        all_targets.append(targets.cpu().numpy())
                        successful_batches += 1
                        
                    except Exception as e:
                        print(f"    ⚠️  Error in batch {batch_idx}/{len(test_loader)}: {e}")
                        # Continue with next batch
                        continue

            if not all_preds:
                print(f"  ❌ No successful predictions for {backbone_name}")
                continue

            print(f"  ✅ Successfully processed {successful_batches}/{len(test_loader)} batches")

            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

            print(f"  📊 Final shapes - Predictions: {all_preds.shape}, Targets: {all_targets.shape}")

            # Calculate metrics with error handling
            try:
                mse = mean_squared_error(all_targets, all_preds)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(all_targets, all_preds)
                r2 = r2_score(all_targets, all_preds)

                # Calculate per-frequency metrics
                freq_metrics = {}
                for i, freq in enumerate(freqs):
                    freq_mse = mean_squared_error(all_targets[:, i], all_preds[:, i])
                    freq_r2 = r2_score(all_targets[:, i], all_preds[:, i])
                    freq_metrics[f"{freq}Hz"] = {"mse": float(freq_mse), "r2": float(freq_r2)}

                results[backbone_name] = {
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2),
                    "predictions": all_preds.tolist(),
                    "targets": all_targets.tolist(),
                    "num_samples": len(all_preds),
                    "successful_batches": successful_batches,
                    "total_batches": len(test_loader),
                    "freq_metrics": freq_metrics
                }

                print(f"  📈 Overall Metrics:")
                print(f"     MSE: {mse:.4f}")
                print(f"     RMSE: {rmse:.4f}")
                print(f"     MAE: {mae:.4f}")
                print(f"     R²: {r2:.4f}")
                
                successful_evaluations += 1

            except Exception as e:
                print(f"  ❌ Error calculating metrics: {e}")
                continue

        except Exception as e:
            print(f"  ❌ Error evaluating {backbone_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    os.makedirs("comparison_results", exist_ok=True)
    with open("comparison_results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n🎉 Successfully evaluated {successful_evaluations}/{len(backbones)} models")

    # Create comparison plots only if we have results
    if results:
        create_comparison_plots(results, freqs)
        print_detailed_analysis(results, freqs)
    else:
        print("❌ No models were successfully evaluated - skipping plots")

    return results


def create_comparison_plots(results, freqs):
    """Create comparison plots for all backbones"""
    
    if not results:
        print("No results to plot")
        return

    # Extract metrics for plotting
    backbones = list(results.keys())
    metrics = ["mse", "rmse", "mae", "r2"]

    print(f"\n📊 Creating plots for {len(backbones)} backbones: {backbones}")

    # Set up plotting style
    plt.style.use('default')
    colors = plt.cm.Set1(np.linspace(0, 1, len(backbones)))

    # Metrics comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = [results[backbone][metric] for backbone in backbones]

        bars = axes[i].bar(backbones, values, color=colors)
        axes[i].set_title(f"{metric.upper()} Comparison", fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric.upper(), fontsize=12)
        axes[i].tick_params(axis="x", rotation=45, labelsize=10)
        axes[i].tick_params(axis="y", labelsize=10)
        axes[i].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if height != 0:  # Only add label if height is not zero
                y_pos = height + 0.01 * (max(values) - min(values)) if max(values) != min(values) else height + 0.01
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

    plt.tight_layout()
    plt.savefig(
        "comparison_results/metrics_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("✅ Metrics comparison plot saved")

    # Per-frequency performance comparison
    create_frequency_performance_plot(results, freqs, backbones)

    # Prediction scatter plots
    create_prediction_scatter_plots(results, freqs, backbones)


def create_frequency_performance_plot(results, freqs, backbones):
    """Create per-frequency R² performance plot"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for backbone in backbones:
        if backbone in results and 'freq_metrics' in results[backbone]:
            freq_r2_values = []
            for freq in freqs:
                freq_key = f"{freq}Hz"
                if freq_key in results[backbone]['freq_metrics']:
                    freq_r2_values.append(results[backbone]['freq_metrics'][freq_key]['r2'])
                else:
                    freq_r2_values.append(0)
            
            ax.plot(freqs, freq_r2_values, marker='o', label=backbone, linewidth=2)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Per-Frequency R² Performance', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Baseline (R²=0)')
    
    plt.tight_layout()
    plt.savefig("comparison_results/frequency_performance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Frequency performance plot saved")


def create_prediction_scatter_plots(results, freqs, backbones):
    """Create prediction vs ground truth scatter plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(backbones)))
    
    for freq_idx, freq in enumerate(freqs):
        plot_created = False
        all_targets_freq = []
        all_preds_freq = []
        
        for backbone, color in zip(backbones, colors):
            if backbone in results:
                try:
                    preds = np.array(results[backbone]["predictions"])[:, freq_idx]
                    targets = np.array(results[backbone]["targets"])[:, freq_idx]
                    
                    axes[freq_idx].scatter(
                        targets, preds, alpha=0.6, label=backbone, 
                        color=color, s=30, edgecolors='white', linewidth=0.5
                    )
                    
                    all_targets_freq.extend(targets)
                    all_preds_freq.extend(preds)
                    plot_created = True
                    
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not plot {backbone} for {freq}Hz: {e}")
                    continue
        
        if plot_created and all_targets_freq:
            # Perfect prediction line
            min_val = min(min(all_targets_freq), min(all_preds_freq))
            max_val = max(max(all_targets_freq), max(all_preds_freq))
            axes[freq_idx].plot([min_val, max_val], [min_val, max_val], 
                               "k--", alpha=0.7, label="Perfect Prediction", linewidth=2)
            
            axes[freq_idx].set_xlabel("Ground Truth RT60 (s)", fontsize=11)
            axes[freq_idx].set_ylabel("Predicted RT60 (s)", fontsize=11)
            axes[freq_idx].set_title(f"{freq} Hz", fontsize=13, fontweight='bold')
            axes[freq_idx].legend(fontsize=9)
            axes[freq_idx].grid(True, alpha=0.3)
            axes[freq_idx].set_aspect('equal', adjustable='box')
        else:
            axes[freq_idx].text(0.5, 0.5, f"No data for {freq}Hz", 
                               ha='center', va='center', transform=axes[freq_idx].transAxes,
                               fontsize=12)
    
    plt.tight_layout()
    plt.savefig("comparison_results/prediction_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Prediction scatter plots saved")


def print_detailed_analysis(results, freqs):
    """Print detailed analysis of results"""
    
    print(f"\n{'='*80}")
    print("DETAILED BACKBONE ANALYSIS")
    print(f"{'='*80}")
    
    if not results:
        print("No results to analyze")
        return
    
    # Overall performance table
    print(f"\n{'OVERALL PERFORMANCE':<20}")
    print(f"{'-'*80}")
    print(f"{'Backbone':<20} {'MSE':<8} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Samples':<8}")
    print(f"{'-'*80}")
    
    # Sort by R² score (descending)
    sorted_backbones = sorted(results.keys(), 
                            key=lambda x: results[x]['r2'], 
                            reverse=True)
    
    for backbone in sorted_backbones:
        r = results[backbone]
        print(f"{backbone:<20} {r['mse']:<8.4f} {r['rmse']:<8.4f} {r['mae']:<8.4f} {r['r2']:<8.4f} {r['num_samples']:<8}")
    
    # Per-frequency analysis
    print(f"\n{'PER-FREQUENCY R² SCORES':<20}")
    print(f"{'-'*80}")
    print(f"{'Backbone':<20}", end="")
    for freq in freqs:
        print(f"{freq:<8}Hz", end="")
    print()
    print(f"{'-'*80}")
    
    for backbone in sorted_backbones:
        if 'freq_metrics' in results[backbone]:
            print(f"{backbone:<20}", end="")
            for freq in freqs:
                freq_key = f"{freq}Hz"
                if freq_key in results[backbone]['freq_metrics']:
                    r2_val = results[backbone]['freq_metrics'][freq_key]['r2']
                    print(f"{r2_val:<10.3f}", end="")
                else:
                    print(f"{'N/A':<10}", end="")
            print()
    
    # Model loading success rates
    print(f"\n{'MODEL LOADING SUCCESS RATES':<20}")
    print(f"{'-'*80}")
    for backbone in sorted_backbones:
        r = results[backbone]
        success_rate = r['successful_batches'] / r['total_batches'] * 100
        print(f"{backbone:<20} {success_rate:<6.1f}% ({r['successful_batches']}/{r['total_batches']} batches)")
    
    print(f"{'='*80}")


def create_summary_table(results, backbones):
    """Create a summary table of results"""
    
    print(f"\n{'='*80}")
    print("BACKBONE EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Backbone':<20} {'MSE':<8} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Samples':<8}")
    print(f"{'-'*80}")
    
    # Sort by R² score (descending)
    sorted_backbones = sorted(backbones, 
                            key=lambda x: results[x]['r2'] if x in results else -999, 
                            reverse=True)
    
    for backbone in sorted_backbones:
        if backbone in results:
            r = results[backbone]
            print(f"{backbone:<20} {r['mse']:<8.4f} {r['rmse']:<8.4f} {r['mae']:<8.4f} {r['r2']:<8.4f} {r['num_samples']:<8}")
        else:
            print(f"{backbone:<20} {'FAILED':<40}")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    results = evaluate_all_backbones()