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
import pandas as pd
from seed import set_seeds


def load_model_safely(model, model_path, device):
    """Load model with compatibility handling"""
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Loaded model successfully with strict=True")
        return True
    except RuntimeError as e:
        print(f"⚠️  Architecture mismatch: {e}")
        try:
            # Filter compatible keys
            model_keys = set(model.state_dict().keys())
            filtered_state_dict = {}
            skipped_keys = []
            
            for key, value in state_dict.items():
                if key in model_keys and value.shape == model.state_dict()[key].shape:
                    filtered_state_dict[key] = value
                else:
                    skipped_keys.append(key)
            
            if skipped_keys:
                print(f"   Skipped {len(skipped_keys)} incompatible keys:")
                for key in skipped_keys[:5]:  # Show first 5 skipped keys
                    print(f"     - {key}")
                if len(skipped_keys) > 5:
                    print(f"     ... and {len(skipped_keys) - 5} more")
            
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            if missing_keys:
                print(f"   Warning: {len(missing_keys)} keys missing (will use random initialization)")
            
            print(f"✅ Loaded {len(filtered_state_dict)} compatible parameters")
            return True
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            return False


def get_model_info(model):
    """Get model architecture information safely"""
    info = {}
    
    # Basic model info
    info['model_class'] = model.__class__.__name__
    
    # Check for feature extractor
    if hasattr(model, 'feature_extractor'):
        info['feature_extractor'] = model.feature_extractor.__class__.__name__
        if hasattr(model.feature_extractor, 'backbone'):
            info['backbone'] = model.feature_extractor.backbone.__class__.__name__
    
    # Check for different types of output layers
    output_layers = []
    for attr_name in ['regression_head', 'classifier', 'fc', 'head', 'output_layer']:
        if hasattr(model, attr_name):
            layer = getattr(model, attr_name)
            output_layers.append(f"{attr_name}: {layer}")
    
    if not output_layers:
        # Look for Sequential or ModuleList attributes that might be output layers
        for name, module in model.named_modules():
            if 'head' in name.lower() or 'fc' in name.lower() or 'classifier' in name.lower():
                output_layers.append(f"{name}: {module}")
    
    info['output_layers'] = output_layers if output_layers else ['No obvious output layer found']
    
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info['total_params'] = total_params
    info['trainable_params'] = trainable_params
    
    return info


def evaluate_dinov2_detailed():
    """Detailed evaluation of DINOv2 model"""
    
    set_seeds(42)
    
    # Model configuration
    backbone_name = "dinov2"
    freqs = [250, 500, 1000, 2000, 4000, 8000]
    model_path = f"comparison_results/{backbone_name}_final.pt"
    
    print(f"{'='*80}")
    print(f"DETAILED DINOV2 MODEL EVALUATION")
    print(f"{'='*80}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        
        # Try alternative paths
        alternative_paths = [
            f"comparison_results/{backbone_name}_synth.pt",
            f"output/{backbone_name}.pt",
            f"checkpoints/{backbone_name}_final.pt"
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"📁 Found alternative model at: {alt_path}")
                model_path = alt_path
                break
        else:
            print(f"❌ No model found in any of the expected locations")
            return None
    
    # Load test dataset
    print("\n📂 Loading test dataset...")
    try:
        test_dataset = ReverbRoomDataset("data/test/real", freqs=freqs, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
        print(f"📊 Found {len(test_dataset)} test samples in {len(test_loader)} batches")
    except Exception as e:
        print(f"❌ Failed to load test dataset: {e}")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load model
    print(f"\n🔧 Loading DINOv2 model...")
    try:
        model = ReverbCNNComparison(
            backbone_name=backbone_name,
            num_frequencies=len(freqs),
            frequencies=freqs,
            dropout_rate=0.15,
        )
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return None
    
    if not load_model_safely(model, model_path, device):
        print("❌ Failed to load DINOv2 model")
        return None
    
    model.to(device)
    model.eval()
    
    # Print model architecture safely
    print(f"\n🏗️  Model Architecture:")
    model_info = get_model_info(model)
    
    print(f"   Model class: {model_info['model_class']}")
    print(f"   Backbone: {backbone_name}")
    
    if 'feature_extractor' in model_info:
        print(f"   Feature extractor: {model_info['feature_extractor']}")
    
    if 'backbone' in model_info:
        print(f"   Backbone implementation: {model_info['backbone']}")
    
    print(f"   Output layers:")
    for layer in model_info['output_layers']:
        print(f"     {layer}")
    
    print(f"   Total parameters: {model_info['total_params']:,}")
    print(f"   Trainable parameters: {model_info['trainable_params']:,}")
    
    # Test single sample
    print(f"\n🧪 Testing single sample...")
    try:
        test_input, test_target = test_dataset[0]
        test_input = test_input.unsqueeze(0).to(device)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {test_output.shape}")
        print(f"   Target shape: {test_target.shape}")
        print(f"   Sample prediction: {test_output.cpu().numpy().flatten()}")
        print(f"   Sample target: {test_target.numpy()}")
        
        # Check if output makes sense
        if test_output.shape[1] != len(freqs):
            print(f"   ⚠️  Output dimension mismatch: expected {len(freqs)}, got {test_output.shape[1]}")
            
    except Exception as e:
        print(f"   ❌ Single sample test failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Run full evaluation
    print(f"\n🔄 Running full evaluation...")
    all_preds = []
    all_targets = []
    batch_errors = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                # Track per-batch metrics
                batch_mse = mean_squared_error(targets.cpu().numpy(), outputs.cpu().numpy())
                print(f"   Batch {batch_idx+1}/{len(test_loader)}: MSE = {batch_mse:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error in batch {batch_idx}: {e}")
                batch_errors.append(batch_idx)
                continue
    
    if not all_preds:
        print("❌ No successful predictions")
        return None
    
    # Combine results
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f"\n📈 Evaluation Results:")
    print(f"   Successful batches: {len(test_loader) - len(batch_errors)}/{len(test_loader)}")
    print(f"   Final prediction shape: {all_preds.shape}")
    print(f"   Final target shape: {all_targets.shape}")
    
    # Check for any obvious issues
    if np.any(np.isnan(all_preds)):
        print("   ⚠️  Warning: NaN values detected in predictions")
    if np.any(np.isinf(all_preds)):
        print("   ⚠️  Warning: Infinite values detected in predictions")
    
    # Calculate detailed metrics
    print(f"\n📊 Computing metrics...")
    
    # Overall metrics
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    print(f"\n🎯 Overall Performance:")
    print(f"   MSE:  {mse:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   R²:   {r2:.6f}")
    
    # Interpretation of R² score
    if r2 < -1:
        print(f"   📊 R² Analysis: Model performs much worse than predicting the mean (R² = {r2:.3f})")
    elif r2 < 0:
        print(f"   📊 R² Analysis: Model performs worse than predicting the mean (R² = {r2:.3f})")
    elif r2 < 0.3:
        print(f"   📊 R² Analysis: Poor performance (R² = {r2:.3f})")
    elif r2 < 0.7:
        print(f"   📊 R² Analysis: Moderate performance (R² = {r2:.3f})")
    else:
        print(f"   📊 R² Analysis: Good performance (R² = {r2:.3f})")
    
    # Per-frequency metrics
    print(f"\n🎵 Per-Frequency Performance:")
    freq_metrics = {}
    for i, freq in enumerate(freqs):
        freq_mse = mean_squared_error(all_targets[:, i], all_preds[:, i])
        freq_rmse = np.sqrt(freq_mse)
        freq_mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        freq_r2 = r2_score(all_targets[:, i], all_preds[:, i])
        
        freq_metrics[freq] = {
            "mse": freq_mse,
            "rmse": freq_rmse,
            "mae": freq_mae,
            "r2": freq_r2,
            "predictions": all_preds[:, i].tolist(),
            "targets": all_targets[:, i].tolist()
        }
        
        print(f"   {freq:>4}Hz: MSE={freq_mse:.4f}, RMSE={freq_rmse:.4f}, MAE={freq_mae:.4f}, R²={freq_r2:.4f}")
    
    # Statistical analysis
    print(f"\n📈 Statistical Analysis:")
    
    # Prediction statistics
    print(f"   Prediction Range: [{all_preds.min():.3f}, {all_preds.max():.3f}]")
    print(f"   Target Range:     [{all_targets.min():.3f}, {all_targets.max():.3f}]")
    print(f"   Prediction Mean:  {all_preds.mean():.3f} ± {all_preds.std():.3f}")
    print(f"   Target Mean:      {all_targets.mean():.3f} ± {all_targets.std():.3f}")
    
    # Check for prediction collapse
    pred_variance = all_preds.var()
    target_variance = all_targets.var()
    print(f"   Prediction Variance: {pred_variance:.6f}")
    print(f"   Target Variance:     {target_variance:.6f}")
    print(f"   Variance Ratio:      {pred_variance/target_variance:.6f}")
    
    if pred_variance < 0.01 * target_variance:
        print("   ⚠️  Warning: Model predictions have very low variance (possible mode collapse)")
    
    # Residual analysis
    residuals = all_targets - all_preds
    print(f"   Residual Mean:    {residuals.mean():.6f}")
    print(f"   Residual Std:     {residuals.std():.6f}")
    print(f"   Residual Range:   [{residuals.min():.3f}, {residuals.max():.3f}]")
    
    # Create detailed visualizations
    create_detailed_plots(all_preds, all_targets, freqs, freq_metrics, residuals)
    
    # Save detailed results
    results = {
        "model_name": "dinov2",
        "model_path": model_path,
        "model_info": model_info,
        "overall_metrics": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        },
        "frequency_metrics": {str(k): v for k, v in freq_metrics.items()},
        "statistics": {
            "num_samples": len(all_preds),
            "prediction_range": [float(all_preds.min()), float(all_preds.max())],
            "target_range": [float(all_targets.min()), float(all_targets.max())],
            "prediction_mean": float(all_preds.mean()),
            "prediction_std": float(all_preds.std()),
            "target_mean": float(all_targets.mean()),
            "target_std": float(all_targets.std()),
            "prediction_variance": float(pred_variance),
            "target_variance": float(target_variance),
            "variance_ratio": float(pred_variance/target_variance),
            "residual_mean": float(residuals.mean()),
            "residual_std": float(residuals.std()),
        },
        "batch_errors": batch_errors
    }
    
    # Save results
    os.makedirs("detailed_evaluation", exist_ok=True)
    with open("detailed_evaluation/dinov2_detailed_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n💾 Detailed results saved to detailed_evaluation/dinov2_detailed_results.json")
    print(f"📊 Plots saved to detailed_evaluation/")
    
    return results


def create_detailed_plots(predictions, targets, freqs, freq_metrics, residuals):
    """Create detailed visualization plots"""
    
    os.makedirs("detailed_evaluation", exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Overall prediction vs target scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.6, s=30)
    
    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Ground Truth RT60 (s)', fontsize=12)
    plt.ylabel('Predicted RT60 (s)', fontsize=12)
    plt.title('DINOv2: Overall Prediction vs Ground Truth', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add R² annotation
    r2_overall = r2_score(targets.flatten(), predictions.flatten())
    plt.text(0.05, 0.95, f'R² = {r2_overall:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("detailed_evaluation/dinov2_overall_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Per-frequency scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, freq in enumerate(freqs):
        freq_preds = predictions[:, i]
        freq_targets = targets[:, i]
        freq_r2 = freq_metrics[freq]['r2']
        
        axes[i].scatter(freq_targets, freq_preds, alpha=0.6, s=40)
        
        # Perfect prediction line
        min_val = min(freq_targets.min(), freq_preds.min())
        max_val = max(freq_targets.max(), freq_preds.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        axes[i].set_xlabel('Ground Truth RT60 (s)', fontsize=11)
        axes[i].set_ylabel('Predicted RT60 (s)', fontsize=11)
        axes[i].set_title(f'{freq} Hz (R² = {freq_r2:.3f})', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        # Make axes equal for better visualization
        all_vals = np.concatenate([freq_targets, freq_preds])
        margin = (all_vals.max() - all_vals.min()) * 0.05
        axes[i].set_xlim(all_vals.min() - margin, all_vals.max() + margin)
        axes[i].set_ylim(all_vals.min() - margin, all_vals.max() + margin)
    
    plt.tight_layout()
    plt.savefig("detailed_evaluation/dinov2_frequency_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Residual analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residual histogram
    axes[0, 0].hist(residuals.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Residuals (Ground Truth - Predicted)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals vs predictions
    axes[0, 1].scatter(predictions.flatten(), residuals.flatten(), alpha=0.6, s=20)
    axes[0, 1].set_xlabel('Predicted RT60 (s)', fontsize=11)
    axes[0, 1].set_ylabel('Residuals', fontsize=11)
    axes[0, 1].set_title('Residuals vs Predictions', fontsize=12, fontweight='bold')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot for residuals
    try:
        from scipy import stats
        stats.probplot(residuals.flatten(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot: Residuals vs Normal Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    except ImportError:
        axes[1, 0].text(0.5, 0.5, 'scipy not available\nfor Q-Q plot', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Q-Q Plot (scipy required)', fontsize=12, fontweight='bold')
    
    # Per-frequency R² scores
    freq_r2_values = [freq_metrics[freq]['r2'] for freq in freqs]
    bars = axes[1, 1].bar(range(len(freqs)), freq_r2_values, color='skyblue', edgecolor='black')
    axes[1, 1].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[1, 1].set_ylabel('R² Score', fontsize=11)
    axes[1, 1].set_title('Per-Frequency R² Performance', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(range(len(freqs)))
    axes[1, 1].set_xticklabels([f'{freq}' for freq in freqs], rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.8, label='Baseline')
    
    # Add value labels on bars
    for bar, value in zip(bars, freq_r2_values):
        height = bar.get_height()
        y_pos = height + 0.01 if height >= 0 else height - 0.03
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=10)
    
    plt.tight_layout()
    plt.savefig("detailed_evaluation/dinov2_residual_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. Error distribution by frequency
    plt.figure(figsize=(12, 8))
    
    freq_errors = []
    freq_labels = []
    
    for freq in freqs:
        freq_residuals = residuals[:, freqs.index(freq)]
        freq_errors.append(np.abs(freq_residuals))
        freq_labels.append(f'{freq}Hz')
    
    # Box plot of absolute errors by frequency
    box_plot = plt.boxplot(freq_errors, labels=freq_labels, patch_artist=True, 
                          showmeans=True, meanline=True)
    
    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(freqs)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Absolute Error (s)', fontsize=12)
    plt.title('DINOv2: Error Distribution by Frequency', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("detailed_evaluation/dinov2_error_by_frequency.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("✅ All detailed plots created and saved")


if __name__ == "__main__":
    set_seeds(42)

    results = evaluate_dinov2_detailed()
    
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"✅ DINOv2 evaluation completed successfully")
        print(f"📊 Overall R²: {results['overall_metrics']['r2']:.6f}")
        
        # Find best and worst frequencies
        freq_r2_scores = {k: v['r2'] for k, v in results['frequency_metrics'].items()}
        best_freq = max(freq_r2_scores, key=freq_r2_scores.get)
        worst_freq = min(freq_r2_scores, key=freq_r2_scores.get)
        
        print(f"📈 Best frequency: {best_freq}Hz (R² = {freq_r2_scores[best_freq]:.4f})")
        print(f"📉 Worst frequency: {worst_freq}Hz (R² = {freq_r2_scores[worst_freq]:.4f})")
        print(f"📁 Results saved in detailed_evaluation/ directory")
        
        # Provide some analysis
        r2 = results['overall_metrics']['r2']
        if r2 < -1:
            print("\n💡 Analysis: The model is performing much worse than simply predicting the mean.")
            print("   This suggests the model hasn't learned meaningful patterns or there are")
            print("   architectural issues with feature extraction or fusion.")
        elif r2 < 0:
            print("\n💡 Analysis: The model performs worse than a naive baseline (predicting mean).")
            print("   This indicates training issues or architectural problems.")
        
        variance_ratio = results['statistics']['variance_ratio']
        if variance_ratio < 0.1:
            print("   The predictions have very low variance compared to targets,")
            print("   suggesting the model may have collapsed to predicting similar values.")
            
    else:
        print("❌ DINOv2 evaluation failed")