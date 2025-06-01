from dataset import ReverbRoomDataset
from model import ReverbCNN
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
import math

from seed import set_seeds


def train_progressive_resolution():
    """Progressive resolution training: start low, increase gradually"""
    set_seeds(42)

    # Define progressive resolution schedule
    resolution_stages = [
        {"resolution": 224, "epochs": 25, "batch_size": 64, "accumulate_grad": 2},
        {"resolution": 320, "epochs": 25, "batch_size": 32, "accumulate_grad": 4},
        {"resolution": 448, "epochs": 30, "batch_size": 16, "accumulate_grad": 8},
    ]

    params = {
        "lr": 0.003,
        "fine_tune_lr": 0.0005,
        "freqs": [250, 500, 1000, 2000, 4000, 8000],
        "synth_model_out": "output/reverbcnn_synth_progressive.pt",
        "final_model_out": "output/reverbcnn_progressive.pt",
        "num_workers": 8,
        "pin_memory": True,
    }

    os.makedirs(os.path.dirname(params["synth_model_out"]), exist_ok=True)

    # Initialize model once
    model = ReverbCNN(num_frequencies=len(params["freqs"]), learning_rate=params["lr"])

    print("=== PROGRESSIVE RESOLUTION TRAINING ON SYNTHETIC DATA ===")

    # Train through each resolution stage
    for stage_idx, stage in enumerate(resolution_stages):
        resolution = stage["resolution"]
        epochs = stage["epochs"]
        batch_size = stage["batch_size"]
        accumulate_grad = stage["accumulate_grad"]

        print(f"\n--- Stage {stage_idx + 1}: {resolution}x{resolution} Resolution ---")
        print(
            f"Epochs: {epochs}, Batch Size: {batch_size}, Grad Accumulation: {accumulate_grad}"
        )
        print(f"Effective Batch Size: {batch_size * accumulate_grad}")

        # Create datasets with current resolution
        synth_train = ReverbRoomDataset(
            "data/train/synth",
            freqs=params["freqs"],
            augment=True,
            image_size=resolution,
        )
        synth_val = ReverbRoomDataset(
            "data/val/synth",
            freqs=params["freqs"],
            augment=False,
            image_size=resolution,
        )

        print(f"Dataset sizes - Train: {len(synth_train)}, Val: {len(synth_val)}")

        # Create data loaders for this stage
        synth_train_loader = DataLoader(
            synth_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=params["num_workers"],
            pin_memory=params["pin_memory"],
            persistent_workers=True,
            drop_last=True,
        )
        synth_val_loader = DataLoader(
            synth_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=params["num_workers"],
            pin_memory=params["pin_memory"],
            persistent_workers=True,
        )

        # Setup callbacks for this stage
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/progressive/stage_{stage_idx + 1}_res{resolution}",
            filename=f"reverbcnn-stage{stage_idx + 1}-res{resolution}-{{epoch:02d}}-{{val_loss:.4f}}",
            save_top_k=2,
            monitor="val_loss",
            mode="min",
            save_weights_only=True,
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=(
                8 if stage_idx < len(resolution_stages) - 1 else 12
            ),  # More patience for final stage
            mode="min",
            min_delta=0.001,
            verbose=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")

        logger = TensorBoardLogger(
            "logs", name=f"progressive_stage{stage_idx + 1}_res{resolution}"
        )

        # Setup trainer for this stage
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            precision="16-mixed",
            accumulate_grad_batches=accumulate_grad,
            gradient_clip_val=1.0,
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
            logger=logger,
            log_every_n_steps=10,
            enable_checkpointing=True,
            enable_progress_bar=True,
            deterministic=False,
            benchmark=True,
        )

        # Train this stage
        trainer.fit(model, synth_train_loader, synth_val_loader)

        # Optionally reduce learning rate between stages
        if stage_idx < len(resolution_stages) - 1:
            for param_group in model.configure_optimizers()["optimizer"].param_groups:
                param_group["lr"] *= 0.8  # Reduce LR by 20% between stages
            print(f"Reduced learning rate for next stage")

    # Save model after progressive synthetic training
    torch.save(model.state_dict(), params["synth_model_out"])
    print(f"\nProgressive synthetic model saved to {params['synth_model_out']}")

    # Fine-tune on real data with highest resolution
    print("\n=== FINE-TUNING ON REAL DATA (High Resolution) ===")

    final_resolution = resolution_stages[-1]["resolution"]  # Use highest resolution
    final_batch_size = resolution_stages[-1]["batch_size"]
    final_accumulate = resolution_stages[-1]["accumulate_grad"]

    print(f"Fine-tuning with {final_resolution}x{final_resolution} resolution")

    # Create fine-tuning model with lower learning rate
    model_finetune = ReverbCNN(
        num_frequencies=len(params["freqs"]), learning_rate=params["fine_tune_lr"]
    )
    model_finetune.load_state_dict(model.state_dict())

    # Create real data datasets
    real_train = ReverbRoomDataset(
        "data/train/real",
        freqs=params["freqs"],
        augment=True,
        image_size=final_resolution,
    )
    real_val = ReverbRoomDataset(
        "data/val/real",
        freqs=params["freqs"],
        augment=False,
        image_size=final_resolution,
    )

    print(f"Real dataset sizes - Train: {len(real_train)}, Val: {len(real_val)}")

    # Create real data loaders
    real_train_loader = DataLoader(
        real_train,
        batch_size=final_batch_size,
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
        persistent_workers=True,
        drop_last=True,
    )
    real_val_loader = DataLoader(
        real_val,
        batch_size=final_batch_size,
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
        persistent_workers=True,
    )

    # Fine-tuning callbacks
    checkpoint_callback_real = ModelCheckpoint(
        dirpath="checkpoints/progressive/finetune",
        filename=f"reverbcnn-finetune-res{final_resolution}-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
    )

    early_stop_callback_real = EarlyStopping(
        monitor="val_loss",
        patience=15,  # More patience for fine-tuning
        mode="min",
        min_delta=0.0005,
        verbose=True,
    )

    logger_real = TensorBoardLogger("logs", name="progressive_finetune")

    # Fine-tuning trainer
    trainer_real = pl.Trainer(
        max_epochs=40,
        accelerator="auto",
        precision="16-mixed",
        accumulate_grad_batches=final_accumulate,
        gradient_clip_val=0.5,  # Lower for fine-tuning
        callbacks=[checkpoint_callback_real, early_stop_callback_real, lr_monitor],
        logger=logger_real,
        log_every_n_steps=5,
        enable_checkpointing=True,
        enable_progress_bar=True,
        deterministic=False,
        benchmark=True,
    )

    trainer_real.fit(model_finetune, real_train_loader, real_val_loader)

    # Save final model
    torch.save(model_finetune.state_dict(), params["final_model_out"])
    print(f"\nFinal progressive model saved to {params['final_model_out']}")

    # Print training summary
    print("\n=== PROGRESSIVE TRAINING COMPLETE ===")
    print("Resolution progression: 224px → 320px → 448px")
    print(f"Final model trained on {final_resolution}x{final_resolution} images")


if __name__ == "__main__":
    train_progressive_resolution()
