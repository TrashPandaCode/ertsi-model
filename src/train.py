from dataset import ReverbRoomDataset
from model import ReverbCNN
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch

from seed import set_seeds


def train():
    set_seeds(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will use: {device}")

    params = {
        "synth_epochs": 100,  # Epochs for training on synthetic data
        "real_epochs": 50,  # Epochs for fine-tuning on real data
        "batch_size": 32,
        "lr": 0.001,
        "fine_tune_lr": 0.0001,  # Lower learning rate for fine-tuning
        "freqs": [250, 500, 1000, 2000, 4000, 8000],
        "synth_model_out": "output/exV4-reverbcnn_synth.pt",  # Checkpoint after synthetic training
        "final_model_out": "output/exV4-reverbcnn.pt",  # Final fine-tuned model
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

    synth_train_loader = DataLoader(
        synth_train, batch_size=params["batch_size"], shuffle=True, num_workers=4
    )
    synth_val_loader = DataLoader(
        synth_val, batch_size=params["batch_size"], shuffle=False, num_workers=4
    )

    real_train_loader = DataLoader(
        real_train, batch_size=params["batch_size"], shuffle=True, num_workers=4
    )
    real_val_loader = DataLoader(
        real_val, batch_size=params["batch_size"], shuffle=False, num_workers=4
    )

    print("\n=== STAGE 1: Training on synthetic data ===")

    model = ReverbCNN(num_frequencies=len(params["freqs"]), learning_rate=params["lr"])

    checkpoint_callback_synth = ModelCheckpoint(
        dirpath="checkpoints/synth",
        filename="exV4-reverbcnn-synth-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback_synth = EarlyStopping(
        monitor="val_loss", patience=5, mode="min"
    )

    logger_synth = TensorBoardLogger("logs", name="exV4-reverbcnn_synth")

    trainer_synth = pl.Trainer(
        max_epochs=params["synth_epochs"],
        accelerator="auto",  # Automatically use GPU if available
        callbacks=[checkpoint_callback_synth, early_stop_callback_synth],
        logger=logger_synth,
        log_every_n_steps=10,
    )

    trainer_synth.fit(model, synth_train_loader, synth_val_loader)

    # Save the model after synthetic training
    torch.save(model.state_dict(), params["synth_model_out"])
    print(f"Synthetic model saved to {params['synth_model_out']}")

    print("\n=== STAGE 2: Fine-tuning on real data ===")

    # Update the learning rate for fine-tuning (lower learning rate)
    model.learning_rate = params["fine_tune_lr"]

    checkpoint_callback_real = ModelCheckpoint(
        dirpath="checkpoints/real",
        filename="exV4-reverbcnn-real-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback_real = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",  # More patience for fine-tuning
    )

    logger_real = TensorBoardLogger("logs", name="exV4-reverbcnn_real_finetune")

    trainer_real = pl.Trainer(
        max_epochs=params["real_epochs"],
        accelerator="auto",
        callbacks=[checkpoint_callback_real, early_stop_callback_real],
        logger=logger_real,
        log_every_n_steps=5,
    )

    trainer_real.fit(model, real_train_loader, real_val_loader)

    # Save the final fine-tuned model
    torch.save(model.state_dict(), params["final_model_out"])
    print(f"Final fine-tuned model saved to {params['final_model_out']}")


if __name__ == "__main__":
    train()
