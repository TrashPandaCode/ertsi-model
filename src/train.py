from dataset import ReverbRoomDataset
from model import ReverbCNN
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch

from seed import set_seeds

def train():
    set_seeds(42)
    
    # Hyperparameters
    params = {
        "epochs": 100,
        "batch_size": 32,
        "lr": 0.001,
        "freqs": [125, 250, 500, 1000, 2000, 4000],
        "model_out": "output/reverbcnn.pt",
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(params["model_out"]), exist_ok=True)
    
    # Load dataset
    dataset = ReverbRoomDataset("data", freqs=params["freqs"])
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=params["batch_size"], 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = ReverbCNN(
        num_frequencies=len(params["freqs"]),
        learning_rate=params["lr"]
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="reverbcnn-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )
    
    # Logger
    logger = TensorBoardLogger("logs", name="reverbcnn")
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=params["epochs"],
        accelerator="auto",  # Automatically use GPU if available
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Save the model
    torch.save(model.state_dict(), params["model_out"])
    print(f"Model saved to {params['model_out']}")


if __name__ == "__main__":
    train()