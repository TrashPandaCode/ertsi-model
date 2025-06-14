import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from complex_non_pre_model import AdvancedReverbCNN
from dataset import ReverbRoomDataset
import torch
import warnings

from seed import set_seeds

warnings.filterwarnings("ignore")


def train():
    # Frequency bands for RT60 prediction
    freqs = [250, 500, 1000, 2000, 4000, 8000]

    # Initialize the advanced model
    model = AdvancedReverbCNN(
        num_frequencies=len(freqs),
        learning_rate=1e-4,  # Lower learning rate for stability
        weight_decay=1e-4,  # L2 regularization
        use_pretrained=False,  # Use ImageNet pretrained weights
        freeze_backbone_epochs=3,  # Freeze backbone for first 3 epochs
    )

    # Create datasets with enhanced augmentation
    train_dataset = ReverbRoomDataset(
        ["data/train/synth/hybrid", "data/train/synth/non-hybrid", "data/train/real"],
        freqs=freqs,
        augment=True,
    )

    val_dataset = ReverbRoomDataset(
        ["data/val/synth", "data/val/real"], freqs=freqs, augment=False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,  # Smaller batch size due to model complexity
        shuffle=True,
        num_workers=8,  # Increase workers for faster data loading
        pin_memory=True,  # Speed up GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True,  # Ensure consistent batch sizes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,  # Larger batch size for validation
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Update model's frequency weights based on training data
    print("Computing frequency-specific weights...")
    model.update_frequency_weights(train_loader)
    print(f"Frequency weights: {model.freq_weights}")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="output/checkpoints",
        filename="advanced_reverbcnn_{epoch:02d}_{val_loss:.4f}",
        save_top_k=3,  # Keep top 3 models
        monitor="val_loss",
        mode="min",
        save_last=True,  # Always save the last checkpoint
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=15,  # Stop if no improvement for 15 epochs
        mode="min",
        verbose=True,
        min_delta=0.001,  # Minimum change to qualify as improvement
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Setup logger
    logger = TensorBoardLogger(
        save_dir="output/logs", name="advanced_reverb_cnn", version=None
    )

    # Configure trainer with advanced settings
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",  # Automatically detect GPU/CPU
        devices="auto",  # Use all available devices
        precision="16-mixed"
        if torch.cuda.is_available()
        else "32",  # Mixed precision for speed
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        gradient_clip_val=1.0,  # Gradient clipping for stability
        accumulate_grad_batches=2,  # Effective batch size = 16 * 2 = 32
        deterministic=False,  # Allow non-deterministic ops for speed
        benchmark=True,  # Optimize cudnn for consistent input sizes
    )

    # Print model info
    print(f"\nModel Summary:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Start training
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)

    # Save the final model
    trainer.save_checkpoint("output/advanced_reverbcnn_final.ckpt")

    # Test the best model
    print("\nLoading best model for final evaluation...")
    best_model = AdvancedReverbCNN.load_from_checkpoint(
        checkpoint_callback.best_model_path, num_frequencies=len(freqs)
    )

    # Run validation on best model
    trainer.validate(best_model, val_loader)

    print(f"\nTraining completed!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"Final model saved to: output/advanced_reverbcnn_final.ckpt")
    print(f"TensorBoard logs: output/logs/advanced_reverb_cnn/")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    set_seeds()
    # Train the model
    train()
