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
from seed import set_seeds


def train_improved_model():
    set_seeds(42)

    # Enhanced parameters
    params = {
        "epochs": 40,
        "batch_size": 24,
        "lr": 0.0005,
        "fine_tune_lr": 0.00005,
        "freqs": [250, 500, 1000, 2000, 4000, 8000],
        "dropout_rate": 0.15,
        "model_out": "output/synth_only_final.pt",
        "warmup_epochs": 8
    }

    #os.makedirs(os.path.dirname(params["synth_model_out"]), exist_ok=True)

    synth_train = ReverbRoomDataset(
        "data/train/synth",
        freqs=params["freqs"],
        augment=True,
    )
    synth_val = ReverbRoomDataset(
        "data/val/synth", freqs=params["freqs"], augment=False
    )

    # Print dataset sizes for information
    print(f"Synth training set: {len(synth_train)} samples")
    print(f"Synth validation set: {len(synth_val)} samples")

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

    # Create and train single model
    print(f"\n{'=' * 60}")
    print("Training Single Model")
    print(f"{'=' * 60}")

    model = ReverbCNN(
        num_frequencies=len(params["freqs"]),
        learning_rate=params["lr"],
        frequencies=params["freqs"],
        dropout_rate=params["dropout_rate"],
        use_scheduler=True,
        warmup_epochs=params["warmup_epochs"],
    )

    # Stage 1: Training on synthetic data
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/synth",
        filename=f"synth-only-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
        verbose=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=12,  # Slightly more patience for improved model
        mode="min",
        verbose=True,
        min_delta=0.0001,  # Minimum change to qualify as improvement
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger = TensorBoardLogger(
        "logs",
        name=f"reverbcnn_synth_only"
    )

    trainer = pl.Trainer(
        max_epochs=params["epochs"],
        accelerator="auto",
        devices="auto",  # Use all available devices
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=5,
        val_check_interval=1.0,  # Validate twice per epoch
        gradient_clip_val=0.5,  # Gradient clipping for stability
        deterministic=True,  # For reproducibility
        enable_model_summary=True,
    )

    trainer.fit(model, synth_train_loader, synth_val_loader)

    # Save model
    os.makedirs(os.path.dirname(params["model_out"]), exist_ok=True)
    torch.save(model.state_dict(), params["model_out"])
    print(f"Final model saved to {params['model_out']}")


if __name__ == "__main__":
    train_improved_model()