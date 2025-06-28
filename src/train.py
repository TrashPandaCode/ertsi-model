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
        "epochs": 200,  # Mehr Epochen für das frühere Early Stopping
        "batch_size": 32,
        "lr": 0.0003,
        "fine_tune_lr": 0.00003,
        "freqs": [250, 500, 1000, 2000, 4000, 8000],
        "model_out": "output/exV3-reverbcnn_synth_only.pt",

    }

    os.makedirs(os.path.dirname(params["model_out"]), exist_ok=True)

    # Nur synthetische Daten verwenden
    synth_train = ReverbRoomDataset(
        "data/train/synth",
        freqs=params["freqs"],
        augment=True,
    )
    synth_val = ReverbRoomDataset(
        "data/val/synth", freqs=params["freqs"], augment=False
    )

    print(f"Synthetic training set: {len(synth_train)} samples")
    print(f"Synthetic validation set: {len(synth_val)} samples")

    synth_train_loader = DataLoader(
        synth_train, batch_size=params["batch_size"], shuffle=True, num_workers=4
    )
    synth_val_loader = DataLoader(
        synth_val, batch_size=params["batch_size"], shuffle=False, num_workers=4
    )

    print("\n=== Training nur auf synthetischen Daten ===")

    model = ReverbCNN(
        num_frequencies=len(params["freqs"]), 
        learning_rate=params["lr"],
        freeze_backbone=True,  # Backbone einfrieren
        freeze_first_layers=True  # Erste Schichten einfrieren
    )

    # Aggressiveres Early Stopping
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/synth_only",
        filename="exV3-reverbcnn-synth-only-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Früheres Early Stopping mit weniger Geduld
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        patience=15,  # Weniger Geduld
        mode="min",
        min_delta=0.001  # Minimale Verbesserung erforderlich
    )

    logger = TensorBoardLogger("logs", name="exV3-reverbcnn_synth_only")

    trainer = pl.Trainer(
        max_epochs=params["epochs"],
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # Gradient Clipping hinzufügen
        gradient_clip_algorithm="norm",
        
    )

    trainer.fit(model, synth_train_loader, synth_val_loader)

    # Modell speichern
    torch.save(model.state_dict(), params["model_out"])
    print(f"Modell gespeichert unter {params['model_out']}")


if __name__ == "__main__":
    train()