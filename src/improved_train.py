from dataset import ReverbRoomDataset
from improved_model import ImprovedReverbCNN, EnsembleReverbCNN
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch
import numpy as np
from seed import set_seeds


def train_improved_model():
    set_seeds(42)

    # Enhanced parameters
    params = {
        "synth_epochs": 80,
        "real_epochs": 40,
        "batch_size": 24,  # Slightly smaller due to more complex model
        "lr": 0.0008,  # Slightly lower learning rate
        "fine_tune_lr": 0.00005,
        "freqs": [250, 500, 1000, 2000, 4000, 8000],
        "dropout_rate": 0.15,
        "warmup_epochs": 8,
        "use_progressive": True,
        "create_ensemble": True,
        "num_ensemble_models": 3,
        "synth_model_out": "output/improved_reverbcnn_synth.pt",
        "final_model_out": "output/improved_reverbcnn.pt",
        "ensemble_out": "output/ensemble_reverbcnn.pt",
    }

    os.makedirs(os.path.dirname(params["synth_model_out"]), exist_ok=True)

    synth_train = ReverbRoomDataset(
        ["data/train/synth/hybrid", "data/train/synth/non-hybrid"],
        freqs=params["freqs"],
        augment=True,
    )
    synth_val = ReverbRoomDataset(
        "data/val/synth", freqs=params["freqs"], augment=False
    )

    real_train = ReverbRoomDataset(
        "data/train/real", freqs=params["freqs"], augment=True
    )
    real_val = ReverbRoomDataset("data/val/real", freqs=params["freqs"], augment=False)

    # Print dataset sizes for information
    print(f"Synthetic training set: {len(synth_train)} samples")
    print(f"Synthetic validation set: {len(synth_val)} samples")
    print(f"Real training set: {len(real_train)} samples")
    print(f"Real validation set: {len(real_val)} samples")

    # Data loaders
    synth_train_loader = DataLoader(
        synth_train,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    synth_val_loader = DataLoader(
        synth_val,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    real_train_loader = DataLoader(
        real_train,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    real_val_loader = DataLoader(
        real_val,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    ensemble_models = []

    for model_idx in range(params["num_ensemble_models"]):
        print(f"\n{'=' * 60}")
        print(
            f"Training Ensemble Model {model_idx + 1}/{params['num_ensemble_models']}"
        )
        print(f"{'=' * 60}")

        # Create model with slight variations for ensemble diversity
        model = ImprovedReverbCNN(
            num_frequencies=len(params["freqs"]),
            learning_rate=params["lr"]
            * (0.8 + 0.4 * np.random.random()),  # Vary LR slightly
            frequencies=params["freqs"],
            dropout_rate=params["dropout_rate"]
            + 0.05 * (np.random.random() - 0.5),  # Vary dropout
            use_scheduler=True,
            warmup_epochs=params["warmup_epochs"],
        )

        # Train individual model
        trained_model = train_single_model(
            model,
            synth_train_loader,
            synth_val_loader,
            real_train_loader,
            real_val_loader,
            params,
            model_suffix=f"_ensemble_{model_idx}",
        )

        ensemble_models.append(trained_model)

    # Create and save ensemble
    print(f"\n{'=' * 60}")
    print("Creating Ensemble Model")
    print(f"{'=' * 60}")

    ensemble = EnsembleReverbCNN(ensemble_models)
    torch.save(ensemble.state_dict(), params["ensemble_out"])
    print(f"Ensemble model saved to {params['ensemble_out']}")


def train_single_model(
    model,
    synth_train_loader,
    synth_val_loader,
    real_train_loader,
    real_val_loader,
    params,
    model_suffix="",
):
    """
    Train a single model with two-stage training: synthetic data first, then fine-tuning on real data.

    Args:
        model: The model to train (ImprovedReverbCNN instance)
        synth_train_loader: DataLoader for synthetic training data
        synth_val_loader: DataLoader for synthetic validation data
        real_train_loader: DataLoader for real training data
        real_val_loader: DataLoader for real validation data
        params: Dictionary containing training parameters
        model_suffix: String suffix for naming checkpoints and logs

    Returns:
        The trained model
    """

    print(f"\n=== STAGE 1: Training on synthetic data{model_suffix} ===")

    # Stage 1: Training on synthetic data
    checkpoint_callback_synth = ModelCheckpoint(
        dirpath=f"checkpoints/synth{model_suffix}",
        filename=f"improved-reverbcnn-synth{model_suffix}-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
        verbose=True,
    )

    early_stop_callback_synth = EarlyStopping(
        monitor="val_loss",
        patience=8,  # Slightly more patience for improved model
        mode="min",
        verbose=True,
        min_delta=0.0001,  # Minimum change to qualify as improvement
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger_synth = TensorBoardLogger(
        "logs",
        name=f"improved-reverbcnn_synth{model_suffix}",
        version=None,  # Auto-increment version
    )

    trainer_synth = pl.Trainer(
        max_epochs=params["synth_epochs"],
        accelerator="auto",
        devices="auto",  # Use all available devices
        callbacks=[checkpoint_callback_synth, early_stop_callback_synth, lr_monitor],
        logger=logger_synth,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        gradient_clip_val=1.0,  # Gradient clipping for stability
        deterministic=True,  # For reproducibility
        enable_model_summary=True,
    )

    # Fit the model on synthetic data
    trainer_synth.fit(model, synth_train_loader, synth_val_loader)

    # Save synthetic model checkpoint
    synth_checkpoint_path = f"output/improved_reverbcnn_synth{model_suffix}.pt"
    os.makedirs(os.path.dirname(synth_checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), synth_checkpoint_path)
    print(f"Synthetic model{model_suffix} saved to {synth_checkpoint_path}")

    print(f"\n=== STAGE 2: Fine-tuning on real data{model_suffix} ===")

    # Stage 2: Fine-tuning on real data
    # Update learning rate for fine-tuning
    original_lr = model.learning_rate
    model.learning_rate = params["fine_tune_lr"]

    # Reset the optimizer with new learning rate
    model.configure_optimizers()

    # Optional: Freeze early layers for fine-tuning (uncomment if desired)
    # for name, param in model.named_parameters():
    #     if 'conv1' in name or 'conv2' in name:
    #         param.requires_grad = False

    checkpoint_callback_real = ModelCheckpoint(
        dirpath=f"checkpoints/real{model_suffix}",
        filename=f"improved-reverbcnn-real{model_suffix}-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
        verbose=True,
    )

    early_stop_callback_real = EarlyStopping(
        monitor="val_loss",
        patience=12,  # More patience for fine-tuning
        mode="min",
        verbose=True,
        min_delta=0.00005,  # Smaller minimum delta for fine-tuning
    )

    logger_real = TensorBoardLogger(
        "logs", name=f"improved-reverbcnn_real_finetune{model_suffix}", version=None
    )

    trainer_real = pl.Trainer(
        max_epochs=params["real_epochs"],
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback_real, early_stop_callback_real, lr_monitor],
        logger=logger_real,
        log_every_n_steps=5,
        val_check_interval=1.0,  # Validate every epoch during fine-tuning
        gradient_clip_val=0.5,  # Lower gradient clipping for fine-tuning
        deterministic=True,
        enable_model_summary=False,  # Already shown in stage 1
    )

    # Fine-tune the model on real data
    trainer_real.fit(model, real_train_loader, real_val_loader)

    # Save final fine-tuned model
    final_checkpoint_path = f"output/improved_reverbcnn_final{model_suffix}.pt"
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final fine-tuned model{model_suffix} saved to {final_checkpoint_path}")

    # Restore original learning rate for consistency
    model.learning_rate = original_lr

    # Get best validation loss for logging
    try:
        best_val_loss = trainer_real.callback_metrics.get("val_loss", "N/A")
        print(f"Best validation loss{model_suffix}: {best_val_loss}")
    except:
        print(f"Could not retrieve best validation loss{model_suffix}")

    return model


if __name__ == "__main__":
    train_improved_model()
