import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from simple_model import SimpleReverbCNN
from dataset import ReverbRoomDataset


def train():
    freqs = [250, 500, 1000, 2000, 4000, 8000]

    model = SimpleReverbCNN(num_frequencies=len(freqs), learning_rate=1e-3)

    train_dataset = ReverbRoomDataset(
        ["data/train/synth/hybrid", "data/train/synth/non-hybrid"],
        freqs=freqs,
        augment=True,
    )
    val_dataset = ReverbRoomDataset("data/val/synth", freqs=freqs, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        callbacks=[ModelCheckpoint(save_top_k=1, monitor="val_loss")],
        log_every_n_steps=5,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("output/simple_reverbcnn.ckpt")


if __name__ == "__main__":
    train()
