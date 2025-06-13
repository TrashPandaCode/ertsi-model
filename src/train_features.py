from dataset_features import RoomAwareReverbDataset
from model_features import RoomSpecificReverbCNN
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch

from seed import set_seeds


def train():
    set_seeds(42)

    params = {
        "synth_epochs": 100,
        "real_epochs": 50,
        "batch_size": 32,
        "lr": 0.001,
        "fine_tune_lr": 0.0001,
        "freqs": [250, 500, 1000, 2000, 4000, 8000],
        "synth_model_out": "output/reverbcnn_synth.pt",
        "final_model_out": "output/reverbcnn.pt",
        "use_room_embeddings": True,
    }

    os.makedirs(os.path.dirname(params["synth_model_out"]), exist_ok=True)

    # Fix: Pass freqs parameter correctly, not use_room_embeddings
    synth_train = RoomAwareReverbDataset(
        "data/train/synth", 
        freqs=params["freqs"],  # Pass the frequency list
        augment=True, 
        include_room_id=params["use_room_embeddings"]
    )
    synth_val = RoomAwareReverbDataset(
        "data/val/synth", 
        freqs=params["freqs"],  # Pass the frequency list
        augment=False, 
        include_room_id=params["use_room_embeddings"]
    )

    real_train = RoomAwareReverbDataset(
        "data/train/real", 
        freqs=params["freqs"],  # Pass the frequency list
        augment=True, 
        include_room_id=params["use_room_embeddings"]
    )
    real_val = RoomAwareReverbDataset(
        "data/val/real", 
        freqs=params["freqs"],  # Pass the frequency list
        augment=False, 
        include_room_id=params["use_room_embeddings"]
    )
    # Calculate total number of unique rooms
    all_rooms = set()
    for dataset in [synth_train, synth_val, real_train, real_val]:
        if hasattr(dataset, 'room_to_id'):
            all_rooms.update(dataset.room_to_id.keys())
    
    num_rooms = len(all_rooms) if params["use_room_embeddings"] else None
    print(f"Total unique rooms: {num_rooms}")

    # Use enhanced model
    model = RoomSpecificReverbCNN(
        num_frequencies=len(params["freqs"]), 
        learning_rate=params["lr"],
        num_rooms=num_rooms
    )

    # Update training loop to handle room IDs
    def custom_collate_fn(batch):
        if len(batch[0]) == 3:  # includes room_id
            images, rt60s, room_ids = zip(*batch)
            return torch.stack(images), torch.stack(rt60s), torch.stack(room_ids)
        else:
            images, rt60s = zip(*batch)
            return torch.stack(images), torch.stack(rt60s), None

    synth_train_loader = DataLoader(
        synth_train, batch_size=params["batch_size"], shuffle=True, 
        num_workers=4, collate_fn=custom_collate_fn
    )
    synth_val_loader = DataLoader(
        synth_val, batch_size=params["batch_size"], shuffle=False, 
        num_workers=4, collate_fn=custom_collate_fn
    )

    real_train_loader = DataLoader(
        real_train, batch_size=params["batch_size"], shuffle=True, 
        num_workers=4, collate_fn=custom_collate_fn
    )
    real_val_loader = DataLoader(
        real_val, batch_size=params["batch_size"], shuffle=False, 
        num_workers=4, collate_fn=custom_collate_fn
    )

    print("\n=== STAGE 1: Training on synthetic data ===")

    model = RoomSpecificReverbCNN(
        num_frequencies=len(params["freqs"]), 
        learning_rate=params["lr"],
        num_rooms=num_rooms
    )

    checkpoint_callback_synth = ModelCheckpoint(
        dirpath="checkpoints/synth",
        filename="reverbcnn-synth-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback_synth = EarlyStopping(
        monitor="val_loss", patience=5, mode="min"
    )

    logger_synth = TensorBoardLogger("logs", name="reverbcnn_synth")

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
    print(f"Synthetic model saved to {params["synth_model_out"]}")

    print("\n=== STAGE 2: Fine-tuning on real data ===")

    # Update the learning rate for fine-tuning (lower learning rate)
    model.learning_rate = params["fine_tune_lr"]

    checkpoint_callback_real = ModelCheckpoint(
        dirpath="checkpoints/real",
        filename="reverbcnn-real-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback_real = EarlyStopping(
        monitor="val_loss", patience=10, mode="min"  # More patience for fine-tuning
    )

    logger_real = TensorBoardLogger("logs", name="reverbcnn_real_finetune")

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
