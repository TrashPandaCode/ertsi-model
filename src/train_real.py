from dataset import ReverbRoomDataset
from model_real import ReverbCNN
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch

from seed import set_seeds


def train():
    set_seeds(42)
    torch.set_float32_matmul_precision('medium')

    params = {
        "epochs": 150,  # Increased epochs since we're only using real data
        "batch_size": 16,  # Smaller batch size for better generalization with limited real data
        "lr": 0.0005,  # Lower initial learning rate for more stable training
        "weight_decay": 1e-3,  # Increased weight decay for regularization
        "freqs": [250, 500, 1000, 2000, 4000, 8000],
        "model_out": "output/reverbcnn_real.pt",
        "accumulate_grad_batches": 2,  # Gradient accumulation to simulate larger batch size
    }

    os.makedirs(os.path.dirname(params["model_out"]), exist_ok=True)

    real_train = ReverbRoomDataset(
        "data/train/real", freqs=params["freqs"], augment=True
    )
    real_val = ReverbRoomDataset("data/val/real", freqs=params["freqs"], augment=False)

    # Print dataset sizes for information
    print(f"Real training set: {len(real_train)} samples")
    print(f"Real validation set: {len(real_val)} samples")

    real_train_loader = DataLoader(
        real_train, 
        batch_size=params["batch_size"], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True  # Drop the last incomplete batch to avoid batch size 1
    )
    real_val_loader = DataLoader(
        real_val, 
        batch_size=params["batch_size"], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False  # Keep all validation samples
    )

    print("\n=== Training on real data ===")

    model = ReverbCNN(
        num_frequencies=len(params["freqs"]), 
        learning_rate=params["lr"],
        weight_decay=params["weight_decay"]
    )

    # Enhanced callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/real",
        filename="reverbcnn-real-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,  # Save the last checkpoint
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        patience=15,  # Increased patience for real data only
        mode="min",
        min_delta=0.001,  # Minimum change to qualify as improvement
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    logger = TensorBoardLogger("logs", name="reverbcnn_real")

    trainer = pl.Trainer(
        max_epochs=params["epochs"],
        accelerator="auto",
        precision="16-mixed",  # Mixed precision for faster training
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=5,
        accumulate_grad_batches=params["accumulate_grad_batches"],
        gradient_clip_val=1.0,  # Gradient clipping for stability
        deterministic=True,  # For reproducible results
        enable_progress_bar=True,
    )

    trainer.fit(model, real_train_loader, real_val_loader)

    # Save the final model
    torch.save(model.state_dict(), params["model_out"])
    print(f"Final model saved to {params['model_out']}")
    
    # Load and save the best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Best model checkpoint: {best_model_path}")
        best_model = ReverbCNN.load_from_checkpoint(
            best_model_path,
            num_frequencies=len(params["freqs"])
        )
        torch.save(best_model.state_dict(), params["model_out"].replace(".pt", "_best.pt"))
        print(f"Best model saved to {params['model_out'].replace('.pt', '_best.pt')}")


if __name__ == "__main__":
    train()