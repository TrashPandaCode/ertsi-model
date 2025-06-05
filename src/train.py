from dataset import ReverbRoomDataset
from model import ReverbCNN
from torch.utils.data import DataLoader, ConcatDataset
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
from datetime import timedelta

from seed import set_seeds


def train_simultaneous():
    """Train on synthetic and real data simultaneously using improved architecture"""
    set_seeds(42)

    # Enhanced parameters with stability fixes
    params = {
        "epochs": 100,
        "batch_size": 24,  # Reduced for stability
        "accumulate_grad_batches": 3,  # Adjusted for effective batch size = 72
        "lr": 0.0005,  # Reduced learning rate for stability
        "freqs": [250, 500, 1000, 2000, 4000, 8000],
        "dropout_rate": 0.15,  # Reduced dropout
        "warmup_epochs": 15,  # Longer warmup
        "model_out": "output/reverbcnn_simultaneous.pt",
        "num_workers": 4,  # Reduced workers to avoid memory issues
        "pin_memory": True,
        "synth_weight": 0.7,
        "real_weight": 0.3,
    }

    os.makedirs(os.path.dirname(params["model_out"]), exist_ok=True)

    # Create datasets
    print("Loading datasets...")
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

    # Check for empty datasets
    if len(synth_train) == 0:
        raise ValueError("Synthetic training dataset is empty!")
    if len(real_train) == 0:
        raise ValueError("Real training dataset is empty!")

    print(f"Synthetic training set: {len(synth_train)} samples")
    print(f"Synthetic validation set: {len(synth_val)} samples")
    print(f"Real training set: {len(real_train)} samples")
    print(f"Real validation set: {len(real_val)} samples")

    # Combine datasets for simultaneous training
    print("\nCombining datasets for simultaneous training...")
    
    # Balance the datasets based on weights
    synth_size = int(len(synth_train) * params["synth_weight"])
    real_size = int(len(real_train) * params["real_weight"])
    
    # Ensure we don't exceed available data
    synth_size = min(synth_size, len(synth_train))
    real_size = min(real_size, len(real_train))
    
    print(f"Using {synth_size} synthetic samples and {real_size} real samples")
    
    # Create subset datasets
    synth_train_subset = torch.utils.data.Subset(
        synth_train, torch.randperm(len(synth_train))[:synth_size]
    )
    real_train_subset = torch.utils.data.Subset(
        real_train, torch.randperm(len(real_train))[:real_size]
    )
    
    # Combine training datasets
    combined_train = ConcatDataset([synth_train_subset, real_train_subset])
    combined_val = ConcatDataset([synth_val, real_val])
    
    print(f"Combined training set: {len(combined_train)} samples")
    print(f"Combined validation set: {len(combined_val)} samples")

    # Calculate steps for scheduler
    effective_batch_size = params["batch_size"] * params["accumulate_grad_batches"]
    steps_per_epoch = math.ceil(len(combined_train) / effective_batch_size)
    total_steps = steps_per_epoch * params["epochs"] + 10

    params["steps_per_epoch"] = steps_per_epoch
    params["total_steps"] = total_steps

    print(f"Calculated steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")

    # Create data loaders with better error handling
    train_loader = DataLoader(
        combined_train,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
        persistent_workers=True,
        drop_last=True,
        # Add timeout to prevent hanging
        timeout=60,
    )
    
    val_loader = DataLoader(
        combined_val,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
        persistent_workers=True,
        timeout=60,
    )

    print("\n=== SIMULTANEOUS TRAINING: Synthetic + Real Data ===")

    # Create improved model with stability fixes
    model = ReverbCNN(
        num_frequencies=len(params["freqs"]),
        learning_rate=params["lr"],
        total_steps=params["total_steps"],
        frequencies=params["freqs"],
        dropout_rate=params["dropout_rate"],
        use_scheduler=True,
        warmup_epochs=params["warmup_epochs"],
    )

    # Enhanced callbacks with NaN detection
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/simultaneous",
        filename="reverbcnn-simultaneous-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,  # Reduced to save disk space
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_weights_only=True,  # Save only weights to reduce size
        verbose=True,
        # Save on train end to avoid NaN issues
        save_on_train_epoch_end=False,
    )

    # More conservative early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,  # More patience
        mode="min",
        min_delta=0.001,  # Larger min_delta
        verbose=True,
        check_finite=True,  # This should catch NaN values
        stopping_threshold=0.01,  # Stop if loss gets very low
    )

    lr_monitor = LearningRateMonitor(
        logging_interval="epoch",
        log_momentum=False,  # Disable momentum logging to reduce complexity
    )

    # Simplified callback for monitoring without potential NaN issues
    class StabilityMonitorCallback(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            # Check for NaN values and log warnings
            current_val_loss = trainer.callback_metrics.get("val_loss", None)
            if current_val_loss is not None:
                if torch.isnan(current_val_loss) or torch.isinf(current_val_loss):
                    print(f"WARNING: Detected {'NaN' if torch.isnan(current_val_loss) else 'Inf'} in validation loss at epoch {trainer.current_epoch}")
                    # Log learning rate for debugging
                    current_lr = trainer.optimizers[0].param_groups[0]['lr']
                    print(f"Current learning rate: {current_lr}")
            
            # Log additional stability metrics
            pl_module.log("epoch", float(trainer.current_epoch), on_epoch=True)

    stability_monitor = StabilityMonitorCallback()

    logger = TensorBoardLogger(
        "logs",
        name="reverbcnn_simultaneous_stable",
        version=None,
    )

    # Enhanced trainer configuration with stability improvements
    trainer = pl.Trainer(
        max_epochs=params["epochs"],
        accelerator="auto",
        devices="auto",
        precision="32",  # Use 32-bit precision instead of 16-mixed for stability
        accumulate_grad_batches=params["accumulate_grad_batches"],
        gradient_clip_val=0.5,  # More aggressive gradient clipping
        gradient_clip_algorithm="norm",  # Use norm-based clipping
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
            stability_monitor,
        ],
        logger=logger,
        log_every_n_steps=20,  # Log less frequently
        val_check_interval=1.0,  # Validate once per epoch instead of twice
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        deterministic=False,  # Allow some non-determinism for stability
        benchmark=False,  # Disable benchmarking to avoid memory issues
        enable_model_summary=True,
        max_time=timedelta(hours=12),
        # Add anomaly detection
        detect_anomaly=True,
        # Limit validation batches for stability
        limit_val_batches=1.0,
    )

    # Add manual validation before training to check for issues
    print("Running initial validation check...")
    try:
        model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                if i >= 2:  # Just check first few batches
                    break
                outputs = model(inputs)
                loss = torch.nn.functional.mse_loss(outputs, targets)
                print(f"Sample validation batch {i}: loss = {loss.item():.6f}")
                
                # Check for NaN/Inf in outputs and targets
                if torch.isnan(outputs).any():
                    print(f"WARNING: NaN detected in model outputs for batch {i}")
                if torch.isnan(targets).any():
                    print(f"WARNING: NaN detected in targets for batch {i}")
                if torch.isinf(outputs).any():
                    print(f"WARNING: Inf detected in model outputs for batch {i}")
                    
        print("Initial validation check completed successfully")
    except Exception as e:
        print(f"ERROR in initial validation: {e}")
        print("This indicates a problem with the model or data")
        return None

    # Train the model
    print("Starting simultaneous training...")
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        print(f"Training interrupted with error: {e}")
        print("Attempting to save current model state...")
        torch.save(model.state_dict(), params["model_out"].replace(".pt", "_interrupted.pt"))

    # Save the final model
    torch.save(model.state_dict(), params["model_out"])
    print(f"Final simultaneous model saved to {params['model_out']}")

    # Print training summary with better error handling
    print("\n=== Simultaneous Training Complete ===")
    try:
        # Get metrics with fallback values
        best_val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float('inf')))
        best_val_mae = trainer.callback_metrics.get("val_mae", torch.tensor(float('inf')))
        best_val_r2 = trainer.callback_metrics.get("val_r2", torch.tensor(-1.0))
        
        # Convert to float safely
        val_loss_str = f"{best_val_loss.item():.4f}" if not torch.isnan(best_val_loss) else "NaN (training instability)"
        val_mae_str = f"{best_val_mae.item():.4f}" if not torch.isnan(best_val_mae) else "NaN (training instability)"
        val_r2_str = f"{best_val_r2.item():.4f}" if not torch.isnan(best_val_r2) else "NaN (training instability)"
        
        print(f"Best validation loss: {val_loss_str}")
        print(f"Best validation MAE: {val_mae_str}")
        print(f"Best validation R²: {val_r2_str}")
        print(f"Training approach: Simultaneous (Synth {params['synth_weight']*100}% + Real {params['real_weight']*100}%)")
        print(f"Total epochs trained: {trainer.current_epoch}")
        
        # Provide debugging hints if NaN occurred
        if torch.isnan(best_val_loss):
            print("\n🔍 NaN DEBUGGING HINTS:")
            print("- Try reducing learning rate further (e.g., 0.0001)")
            print("- Use stronger gradient clipping (e.g., 0.1)")
            print("- Check for corrupted data samples")
            print("- Consider using different loss function")
            print("- Ensure proper data normalization")
        
    except Exception as e:
        print(f"Could not retrieve metrics: {e}")

    return model


def train_stable_baseline():
    """Simplified stable training approach for debugging"""
    set_seeds(42)
    
    # Very conservative parameters
    params = {
        "epochs": 50,
        "batch_size": 16,
        "accumulate_grad_batches": 4,
        "lr": 0.0001,  # Very low learning rate
        "freqs": [250, 500, 1000, 2000, 4000, 8000],
        "dropout_rate": 0.1,  # Low dropout
        "warmup_epochs": 20,  # Long warmup
        "model_out": "output/reverbcnn_stable_baseline.pt",
        "num_workers": 2,
        "pin_memory": False,  # Disable for stability
    }
    
    # Use only synthetic data for baseline
    print("Loading synthetic data only for stable baseline...")
    train_dataset = ReverbRoomDataset(
        "data/train/synth", freqs=params["freqs"], augment=False  # No augmentation
    )
    val_dataset = ReverbRoomDataset(
        "data/val/synth", freqs=params["freqs"], augment=False
    )
    
    if len(train_dataset) == 0:
        print("No training data found!")
        return None
    
    print(f"Training on {len(train_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
    )
    
    # Simple model without complex features
    model = ReverbCNN(
        num_frequencies=len(params["freqs"]),
        learning_rate=params["lr"],
        frequencies=params["freqs"],
        dropout_rate=params["dropout_rate"],
        use_scheduler=False,  # No scheduler for simplicity
    )
    
    # Minimal callbacks
    trainer = pl.Trainer(
        max_epochs=params["epochs"],
        precision="32",  # Full precision
        gradient_clip_val=0.1,
        enable_checkpointing=False,  # Disable for simplicity
        enable_progress_bar=True,
        log_every_n_steps=50,
        val_check_interval=1.0,
    )
    
    print("Starting stable baseline training...")
    trainer.fit(model, train_loader, val_loader)
    
    torch.save(model.state_dict(), params["model_out"])
    print(f"Stable baseline model saved to {params['model_out']}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ReverbCNN with stability fixes")
    parser.add_argument("--mode", choices=["stable", "baseline"], default="stable",
                       help="Training mode: stable (improved) or baseline (simple)")
    
    args = parser.parse_args()
    
    if args.mode == "stable":
        print("Training with stability improvements...")
        train_simultaneous()
    else:
        print("Training stable baseline for debugging...")
        train_stable_baseline()