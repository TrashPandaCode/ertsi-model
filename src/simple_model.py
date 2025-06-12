import torch
from torch import nn
import pytorch_lightning as pl


class SimpleReverbCNN(pl.LightningModule):
    def __init__(self, num_frequencies=6, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, self.hparams.num_frequencies)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
