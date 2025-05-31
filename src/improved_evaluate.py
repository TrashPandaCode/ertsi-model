from dataset import ReverbRoomDataset
from improved_model import EnsembleReverbCNN
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

from seed import set_seeds


def evaluate_ensemble(
    model_path="output/ensemble_reverbcnn.pt",
    individual_model_paths=None,
    data_dir="data/test/real",
    batch_size=32,
    mc_samples=50,
    save_individual_predictions=True,
):
    """
    Comprehensive evaluation of ensemble reverb model with uncertainty quantification.

    Args:
        model_path: Path to the ensemble model
        individual_model_paths: List of paths to individual models (optional)
        data_dir: Directory containing test data
        batch_size: Batch size for evaluation
        mc_samples: Number of Monte Carlo samples for uncertainty estimation
        save_individual_predictions: Whether to save predictions from individual models
    """
    set_seeds(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    freqs = [250, 500, 1000, 2000, 4000, 8000]

    # Load test dataset
    dataset = ReverbRoomDataset(data_dir, freqs=freqs, augment=False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Test dataset size: {len(dataset)} samples")

    # Load ensemble model
    print(f"Loading ensemble model from: {model_path}")
    ensemble_model = EnsembleReverbCNN([])  # Initialize empty ensemble
    ensemble_state = torch.load(model_path, map_location=device)
    ensemble_model.load_state_dict(ensemble_state)
    ensemble_model.to(device)
    ensemble_model.eval()

    num_models = len(ensemble_model.models)
    print(f"Ensemble contains {num_models} models")

    # Initialize storage for results
    all_ensemble_preds = []
    all_individual_preds = []
    all_targets = []
    all_ensemble_uncertainties = []

    # Evaluate ensemble model
    print("\nEvaluating ensemble model...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Get ensemble predictions and individual model predictions
            ensemble_output, individual_outputs = (
                ensemble_model.forward_with_individual(inputs)
            )

            # Store results
            all_ensemble_preds.append(ensemble_output.cpu().numpy())
            all_individual_preds.append(individual_outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Calculate ensemble uncertainty (std across models)
            uncertainty = torch.std(individual_outputs, dim=1)
            all_ensemble_uncertainties.append(uncertainty.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    # Concatenate all batches
    all_ensemble_preds = np.concatenate(all_ensemble_preds, axis=0)
    all_individual_preds = np.concatenate(all_individual_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_ensemble_uncertainties = np.concatenate(all_ensemble_uncertainties, axis=0)

    print(
        f"Final shapes - Ensemble: {all_ensemble_preds.shape}, Individual: {all_individual_preds.shape}"
    )

    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(
        all_ensemble_preds,
        all_individual_preds,
        all_targets,
        all_ensemble_uncertainties,
        freqs,
        num_models,
    )

    # Print results
    print_evaluation_results(metrics, freqs, num_models)

    # Create output directories
    os.makedirs("evaluation", exist_ok=True)
    os.makedirs("evaluation/plots", exist_ok=True)

    # Save metrics
    with open("evaluation/ensemble_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Generate comprehensive plots
    generate_ensemble_plots(
        all_ensemble_preds,
        all_individual_preds,
        all_targets,
        all_ensemble_uncertainties,
        freqs,
        num_models,
        metrics,
    )

    # Perform Monte Carlo Dropout if available
    if hasattr(ensemble_model.models[0], "enable_dropout"):
        print(f"\nPerforming Monte Carlo Dropout with {mc_samples} samples...")
        mc_results = perform_mc_dropout_evaluation(
            ensemble_model, dataloader, device, mc_samples, freqs
        )

        # Save MC results
        with open("evaluation/mc_dropout_results.json", "w") as f:
            json.dump(mc_results, f, indent=4)

        # Plot MC results
        plot_mc_dropout_results(mc_results, freqs)

    # Save individual model predictions if requested
    if save_individual_predictions:
        np.savez(
            "evaluation/predictions.npz",
            ensemble_preds=all_ensemble_preds,
            individual_preds=all_individual_preds,
            targets=all_targets,
            uncertainties=all_ensemble_uncertainties,
            frequencies=freqs,
        )

    print(f"\nEvaluation complete. Results saved to 'evaluation/' directory.")
    return metrics


def calculate_comprehensive_metrics(
    ensemble_preds, individual_preds, targets, uncertainties, freqs, num_models
):
    """Calculate comprehensive metrics for ensemble evaluation."""
    metrics = {
        "ensemble": {"frequency": {}, "overall": {}},
        "individual_models": {},
        "uncertainty": {"frequency": {}, "overall": {}},
        "ensemble_stats": {},
    }

    # Ensemble overall metrics
    mse = mean_squared_error(targets, ensemble_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, ensemble_preds)
    r2 = r2_score(targets, ensemble_preds)

    metrics["ensemble"]["overall"] = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

    # Ensemble per-frequency metrics
    for i, freq in enumerate(freqs):
        freq_preds = ensemble_preds[:, i]
        freq_targets = targets[:, i]
        freq_uncertainty = uncertainties[:, i]

        freq_mse = mean_squared_error(freq_targets, freq_preds)
        freq_rmse = np.sqrt(freq_mse)
        freq_mae = mean_absolute_error(freq_targets, freq_preds)
        freq_r2 = r2_score(freq_targets, freq_preds)

        metrics["ensemble"]["frequency"][str(freq)] = {
            "mse": float(freq_mse),
            "rmse": float(freq_rmse),
            "mae": float(freq_mae),
            "r2": float(freq_r2),
        }

        # Uncertainty metrics
        metrics["uncertainty"]["frequency"][str(freq)] = {
            "mean_uncertainty": float(np.mean(freq_uncertainty)),
            "std_uncertainty": float(np.std(freq_uncertainty)),
            "median_uncertainty": float(np.median(freq_uncertainty)),
            "max_uncertainty": float(np.max(freq_uncertainty)),
        }

    # Individual model metrics
    for model_idx in range(num_models):
        model_preds = individual_preds[:, model_idx, :]

        model_mse = mean_squared_error(targets, model_preds)
        model_rmse = np.sqrt(model_mse)
        model_mae = mean_absolute_error(targets, model_preds)
        model_r2 = r2_score(targets, model_preds)

        metrics["individual_models"][f"model_{model_idx}"] = {
            "mse": float(model_mse),
            "rmse": float(model_rmse),
            "mae": float(model_mae),
            "r2": float(model_r2),
        }

    # Ensemble statistics
    individual_mses = [
        metrics["individual_models"][f"model_{i}"]["mse"] for i in range(num_models)
    ]
    ensemble_improvement = (
        (np.mean(individual_mses) - metrics["ensemble"]["overall"]["mse"])
        / np.mean(individual_mses)
        * 100
    )

    metrics["ensemble_stats"] = {
        "num_models": num_models,
        "individual_model_mse_mean": float(np.mean(individual_mses)),
        "individual_model_mse_std": float(np.std(individual_mses)),
        "ensemble_improvement_pct": float(ensemble_improvement),
        "mean_uncertainty_overall": float(np.mean(uncertainties)),
    }

    # Overall uncertainty metrics
    metrics["uncertainty"]["overall"] = {
        "mean_uncertainty": float(np.mean(uncertainties)),
        "std_uncertainty": float(np.std(uncertainties)),
        "median_uncertainty": float(np.median(uncertainties)),
    }

    return metrics


def print_evaluation_results(metrics, freqs, num_models):
    """Print comprehensive evaluation results."""
    print("\n" + "=" * 80)
    print("ENSEMBLE MODEL EVALUATION RESULTS")
    print("=" * 80)

    print(f"\nEnsemble Configuration:")
    print(f"  Number of models: {num_models}")
    print(f"  Test frequencies: {freqs}")

    print(f"\n=== ENSEMBLE OVERALL PERFORMANCE ===")
    ensemble_overall = metrics["ensemble"]["overall"]
    print(f"MSE:  {ensemble_overall['mse']:.6f}")
    print(f"RMSE: {ensemble_overall['rmse']:.6f}")
    print(f"MAE:  {ensemble_overall['mae']:.6f}")
    print(f"R²:   {ensemble_overall['r2']:.6f}")

    print(f"\n=== INDIVIDUAL MODEL COMPARISON ===")
    individual_mses = []
    for i in range(num_models):
        model_metrics = metrics["individual_models"][f"model_{i}"]
        individual_mses.append(model_metrics["mse"])
        print(
            f"Model {i + 1}: MSE={model_metrics['mse']:.6f}, R²={model_metrics['r2']:.6f}"
        )

    ensemble_stats = metrics["ensemble_stats"]
    print(f"\nEnsemble vs Individual:")
    print(
        f"  Individual MSE (mean ± std): {ensemble_stats['individual_model_mse_mean']:.6f} ± {ensemble_stats['individual_model_mse_std']:.6f}"
    )
    print(f"  Ensemble MSE: {ensemble_overall['mse']:.6f}")
    print(f"  Improvement: {ensemble_stats['ensemble_improvement_pct']:.2f}%")

    print(f"\n=== UNCERTAINTY ANALYSIS ===")
    uncertainty_overall = metrics["uncertainty"]["overall"]
    print(f"Mean uncertainty: {uncertainty_overall['mean_uncertainty']:.6f}")
    print(f"Std uncertainty:  {uncertainty_overall['std_uncertainty']:.6f}")

    print(f"\n=== PERFORMANCE BY FREQUENCY ===")
    for freq in freqs:
        freq_metrics = metrics["ensemble"]["frequency"][str(freq)]
        freq_uncertainty = metrics["uncertainty"]["frequency"][str(freq)]
        print(f"\n{freq} Hz:")
        print(f"  MSE: {freq_metrics['mse']:.6f}, R²: {freq_metrics['r2']:.6f}")
        print(
            f"  Uncertainty: {freq_uncertainty['mean_uncertainty']:.6f} ± {freq_uncertainty['std_uncertainty']:.6f}"
        )


def generate_ensemble_plots(
    ensemble_preds, individual_preds, targets, uncertainties, freqs, num_models, metrics
):
    """Generate comprehensive plots for ensemble evaluation."""

    # 1. Ensemble predictions vs ground truth
    plt.figure(figsize=(18, 12))
    for i, freq in enumerate(freqs):
        plt.subplot(2, 3, i + 1)
        plt.scatter(
            targets[:, i],
            ensemble_preds[:, i],
            alpha=0.6,
            c=uncertainties[:, i],
            cmap="viridis",
            s=20,
        )

        # Perfect prediction line
        min_val = min(targets[:, i].min(), ensemble_preds[:, i].min())
        max_val = max(targets[:, i].max(), ensemble_preds[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

        plt.title(f"Ensemble Predictions @ {freq} Hz")
        plt.xlabel("Ground Truth RT60 (s)")
        plt.ylabel("Predicted RT60 (s)")

        # Add colorbar for uncertainty
        cbar = plt.colorbar()
        cbar.set_label("Uncertainty (std)")

        # Add metrics
        freq_metrics = metrics["ensemble"]["frequency"][str(freq)]
        plt.text(
            0.05,
            0.95,
            f"R²: {freq_metrics['r2']:.4f}\nRMSE: {freq_metrics['rmse']:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(
        "evaluation/plots/ensemble_predictions_vs_truth.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Individual model comparison
    plt.figure(figsize=(20, 12))
    for i, freq in enumerate(freqs):
        plt.subplot(2, 3, i + 1)

        # Plot individual models
        for model_idx in range(num_models):
            plt.scatter(
                targets[:, i],
                individual_preds[:, model_idx, i],
                alpha=0.3,
                s=10,
                label=f"Model {model_idx + 1}",
            )

        # Plot ensemble
        plt.scatter(
            targets[:, i],
            ensemble_preds[:, i],
            alpha=0.8,
            s=15,
            color="red",
            label="Ensemble",
            marker="x",
        )

        # Perfect prediction line
        min_val = targets[:, i].min()
        max_val = targets[:, i].max()
        plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2)

        plt.title(f"Model Comparison @ {freq} Hz")
        plt.xlabel("Ground Truth RT60 (s)")
        plt.ylabel("Predicted RT60 (s)")
        if i == 0:  # Only show legend for first subplot
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(
        "evaluation/plots/individual_vs_ensemble.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. Uncertainty distribution
    plt.figure(figsize=(18, 12))
    for i, freq in enumerate(freqs):
        plt.subplot(2, 3, i + 1)
        plt.hist(uncertainties[:, i], bins=50, alpha=0.7, density=True)
        plt.title(f"Uncertainty Distribution @ {freq} Hz")
        plt.xlabel("Prediction Uncertainty (std)")
        plt.ylabel("Density")

        # Add statistics
        mean_unc = np.mean(uncertainties[:, i])
        std_unc = np.std(uncertainties[:, i])
        plt.axvline(
            mean_unc, color="red", linestyle="--", label=f"Mean: {mean_unc:.4f}"
        )
        plt.axvline(
            mean_unc + std_unc,
            color="orange",
            linestyle=":",
            label=f"±1σ: {std_unc:.4f}",
        )
        plt.axvline(mean_unc - std_unc, color="orange", linestyle=":")
        plt.legend()

    plt.tight_layout()
    plt.savefig(
        "evaluation/plots/uncertainty_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 4. Error vs Uncertainty correlation
    plt.figure(figsize=(18, 12))
    for i, freq in enumerate(freqs):
        plt.subplot(2, 3, i + 1)
        errors = np.abs(ensemble_preds[:, i] - targets[:, i])

        plt.scatter(uncertainties[:, i], errors, alpha=0.6, s=20)

        # Calculate correlation
        correlation = np.corrcoef(uncertainties[:, i], errors)[0, 1]

        plt.title(f"Error vs Uncertainty @ {freq} Hz\nCorrelation: {correlation:.3f}")
        plt.xlabel("Prediction Uncertainty")
        plt.ylabel("Absolute Error")

        # Add trend line
        z = np.polyfit(uncertainties[:, i], errors, 1)
        p = np.poly1d(z)
        plt.plot(uncertainties[:, i], p(uncertainties[:, i]), "r--", alpha=0.8)

    plt.tight_layout()
    plt.savefig(
        "evaluation/plots/error_vs_uncertainty.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 5. Model agreement heatmap
    plt.figure(figsize=(12, 8))

    # Calculate pairwise correlations between models
    correlations = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(num_models):
            pred_i = individual_preds[:, i, :].flatten()
            pred_j = individual_preds[:, j, :].flatten()
            correlations[i, j] = np.corrcoef(pred_i, pred_j)[0, 1]

    sns.heatmap(
        correlations,
        annot=True,
        cmap="coolwarm",
        center=0,
        xticklabels=[f"Model {i + 1}" for i in range(num_models)],
        yticklabels=[f"Model {i + 1}" for i in range(num_models)],
    )
    plt.title("Model Prediction Correlations")
    plt.tight_layout()
    plt.savefig("evaluation/plots/model_correlations.png", dpi=300, bbox_inches="tight")
    plt.close()


def perform_mc_dropout_evaluation(
    ensemble_model, dataloader, device, mc_samples, freqs
):
    """Perform Monte Carlo dropout evaluation for additional uncertainty estimation."""
    print("Enabling dropout for MC sampling...")

    # Enable dropout for all models in ensemble
    for model in ensemble_model.models:
        if hasattr(model, "enable_dropout"):
            model.enable_dropout()

    mc_predictions = []
    targets = []

    with torch.no_grad():
        for sample_idx in range(mc_samples):
            sample_preds = []
            sample_targets = []

            for inputs, batch_targets in dataloader:
                inputs = inputs.to(device)
                outputs = ensemble_model(inputs)
                sample_preds.append(outputs.cpu().numpy())

                if sample_idx == 0:  # Only collect targets once
                    sample_targets.append(batch_targets.numpy())

            mc_predictions.append(np.concatenate(sample_preds, axis=0))
            if sample_idx == 0:
                targets = np.concatenate(sample_targets, axis=0)

    mc_predictions = np.stack(
        mc_predictions, axis=0
    )  # [mc_samples, n_examples, n_freqs]
    mc_means = np.mean(mc_predictions, axis=0)
    mc_stds = np.std(mc_predictions, axis=0)

    # Calculate MC dropout metrics
    mc_results = {
        "mc_samples": mc_samples,
        "epistemic_uncertainty": {},
        "predictive_performance": {},
    }

    # Overall MC metrics
    mc_mse = mean_squared_error(targets, mc_means)
    mc_rmse = np.sqrt(mc_mse)
    mc_mae = mean_absolute_error(targets, mc_means)

    mc_results["predictive_performance"]["overall"] = {
        "mse": float(mc_mse),
        "rmse": float(mc_rmse),
        "mae": float(mc_mae),
        "mean_epistemic_uncertainty": float(np.mean(mc_stds)),
    }

    # Per-frequency MC metrics
    for i, freq in enumerate(freqs):
        freq_uncertainty = mc_stds[:, i]
        mc_results["epistemic_uncertainty"][str(freq)] = {
            "mean": float(np.mean(freq_uncertainty)),
            "std": float(np.std(freq_uncertainty)),
            "median": float(np.median(freq_uncertainty)),
            "percentile_95": float(np.percentile(freq_uncertainty, 95)),
        }

    return mc_results


def plot_mc_dropout_results(mc_results, freqs):
    """Plot Monte Carlo dropout results."""
    # This would require the actual MC predictions data
    # For now, just plot the uncertainty statistics
    pass


if __name__ == "__main__":
    evaluate_ensemble()
