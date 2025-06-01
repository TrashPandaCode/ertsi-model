from dataset import ReverbRoomDataset
from model import ReverbCNN
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch

from seed import set_seeds


def train():
    set_seeds(42)

    params = {
        "synth_epochs": 80,  # Reduced with better scheduling
        "real_epochs": 40,  # Reduced with better fine-tuning
        "batch_size": 64,  # Increased batch size
        "accumulate_grad_batches": 2,  # Effective batch size = 128
        "lr": 0.003,  # Higher initial LR for OneCycleLR
        "fine_tune_lr": 0.0005,  # Adjusted for fine-tuning
        "freqs": [250, 500, 1000, 2000, 4000, 8000],
        "synth_model_out": "output/reverbcnn_synth.pt",
        "final_model_out": "output/reverbcnn.pt",
        "num_workers": 8,  # Increased workers
        "pin_memory": True,  # Pin memory for GPU
    }

    os.makedirs(os.path.dirname(params["synth_model_out"]), exist_ok=True)

    # Create datasets
    synth_train = ReverbRoomDataset(
        "data/train/synth", freqs=params["freqs"], augment=True
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

    # Calculate steps per epoch more accurately
    effective_batch_size = params["batch_size"] * params["accumulate_grad_batches"]
    import math

    # Use math.ceil to ensure we don't underestimate the number of steps
    synth_steps_per_epoch = math.ceil(len(synth_train) / effective_batch_size)
    real_steps_per_epoch = math.ceil(len(real_train) / effective_batch_size)

    # Add buffer to total steps to account for any rounding issues
    synth_total_steps = synth_steps_per_epoch * params["synth_epochs"] + 10
    real_total_steps = real_steps_per_epoch * params["real_epochs"] + 10

    # Add these to params for the model
    params["synth_steps_per_epoch"] = synth_steps_per_epoch
    params["synth_total_steps"] = synth_total_steps
    params["real_steps_per_epoch"] = real_steps_per_epoch
    params["real_total_steps"] = real_total_steps

    print(f"Calculated steps per epoch - Synth: {synth_steps_per_epoch}, Real: {real_steps_per_epoch}")
    print(f"Total steps - Synth: {synth_total_steps}, Real: {real_total_steps}")


    # Optimized data loaders
    synth_train_loader = DataLoader(
        synth_train,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True,  # For consistent batch sizes
    )
    synth_val_loader = DataLoader(
        synth_val,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
        persistent_workers=True,
    )

    real_train_loader = DataLoader(
        real_train,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
        persistent_workers=True,
        drop_last=True,
    )
    real_val_loader = DataLoader(
        real_val,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
        persistent_workers=True,
    )

    print("\n=== STAGE 1: Training on synthetic data ===")

    # Create model with total steps for synthetic training
    model = ReverbCNN(
        num_frequencies=len(params["freqs"]),
        learning_rate=params["lr"],
        total_steps=params["synth_total_steps"],
    )

    # Enhanced callbacks
    checkpoint_callback_synth = ModelCheckpoint(
        dirpath="checkpoints/synth",
        filename="reverbcnn-synth-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,  # Faster saving
    )

    early_stop_callback_synth = EarlyStopping(
        monitor="val_loss",
        patience=8,  # Increased patience
        mode="min",
        min_delta=0.001,  # Minimum change to qualify as improvement
        verbose=True,
    )

    # Add learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger_synth = TensorBoardLogger("logs", name="reverbcnn_synth_optimized")

    # Optimized trainer for synthetic data
    trainer_synth = pl.Trainer(
        max_epochs=params["synth_epochs"],
        accelerator="auto",
        precision="16-mixed",  # Mixed precision for speed
        accumulate_grad_batches=params["accumulate_grad_batches"],
        gradient_clip_val=1.0,  # Gradient clipping
        callbacks=[checkpoint_callback_synth, early_stop_callback_synth, lr_monitor],
        logger=logger_synth,
        log_every_n_steps=5,
        enable_checkpointing=True,
        enable_progress_bar=True,
        deterministic=False,  # Allow non-determinism for speed
        benchmark=True,  # Optimize for consistent input sizes
    )

    trainer_synth.fit(model, synth_train_loader, synth_val_loader)

    # Save the model after synthetic training
    torch.save(model.state_dict(), params["synth_model_out"])
    print(f"Synthetic model saved to {params['synth_model_out']}")

    print("\n=== STAGE 2: Fine-tuning on real data ===")

    # Create a new model instance for fine-tuning with different LR and steps
    model_finetune = ReverbCNN(
        num_frequencies=len(params["freqs"]),
        learning_rate=params["fine_tune_lr"],
        total_steps=params["real_total_steps"],
    )

    # Load the synthetic weights
    model_finetune.load_state_dict(model.state_dict())

    # Optionally freeze backbone for initial fine-tuning epochs
    # Uncomment to freeze backbone initially
    # for param in model_finetune.backbone.parameters():
    #     param.requires_grad = False

    checkpoint_callback_real = ModelCheckpoint(
        dirpath="checkpoints/real",
        filename="reverbcnn-real-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
    )

    early_stop_callback_real = EarlyStopping(
        monitor="val_loss",
        patience=12,  # More patience for fine-tuning
        mode="min",
        min_delta=0.0005,  # Smaller delta for fine-tuning
        verbose=True,
    )

    logger_real = TensorBoardLogger("logs", name="reverbcnn_real_finetune_optimized")

    # Optimized trainer for real data fine-tuning
    trainer_real = pl.Trainer(
        max_epochs=params["real_epochs"],
        accelerator="auto",
        precision="16-mixed",  # Mixed precision
        accumulate_grad_batches=params["accumulate_grad_batches"],
        gradient_clip_val=0.5,  # Lower gradient clipping for fine-tuning
        callbacks=[checkpoint_callback_real, early_stop_callback_real, lr_monitor],
        logger=logger_real,
        log_every_n_steps=3,  # More frequent logging for fine-tuning
        enable_checkpointing=True,
        enable_progress_bar=True,
        deterministic=False,
        benchmark=True,
    )

    trainer_real.fit(model_finetune, real_train_loader, real_val_loader)

    # Save the final fine-tuned model
    torch.save(model_finetune.state_dict(), params["final_model_out"])
    print(f"Final fine-tuned model saved to {params['final_model_out']}")

    # Print training summary
    print("\n=== Training Complete ===")
    print(
        f"Best synthetic validation loss: {trainer_synth.callback_metrics.get('val_loss', 'N/A')}"
    )
    print(
        f"Best real validation loss: {trainer_real.callback_metrics.get('val_loss', 'N/A')}"
    )


def train_with_validation_monitoring():
    """Enhanced training with additional validation monitoring"""
    set_seeds(42)

    # Add validation frequency monitoring
    class ValidationMetricsCallback(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            # Log per-frequency validation metrics
            if hasattr(pl_module, "last_val_outputs"):
                for i in range(len(pl_module.hparams.freqs)):
                    freq_mae = pl_module.last_val_freq_maes[i]
                    pl_module.log(
                        f"val_mae_freq_{pl_module.hparams.freqs[i]}Hz", freq_mae
                    )

    # Use the enhanced training function
    train()


if __name__ == "__main__":
    train()
