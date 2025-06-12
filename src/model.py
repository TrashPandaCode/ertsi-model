import torch
from torch import nn
import pytorch_lightning as pl
import load_model
from torchvision import models


class ReverbCNN(pl.LightningModule):
    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def __init__(self, num_frequencies=6, learning_rate=0.001):
        super().__init__()

        self.save_hyperparameters()
        self.loss_fn = nn.HuberLoss(delta=1.0)

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2)
            )

        def up_block(in_ch, out_ch):
            return conv_block(in_ch, out_ch)

        # Downsampling path
        self.enc1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 512)

        # Upsampling path
        self.up4 = up_block(1024, 512)
        self.up3 = up_block(512 + 256, 256)   # 768 -> 256
        self.up2 = up_block(256 + 128, 128)   # 384 -> 128
        self.up1 = up_block(128 + 64, 64)     # 192 -> 64

        # Final conv to reduce to single-channel feature map
        self.final_conv = nn.Conv2d(64, 32, kernel_size=1)

        # Global Average Pooling -> FC for regression
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_frequencies)

    def forward(self, x):
        # Down
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        # Up
        b_up = nn.functional.interpolate(b, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = self.up4(torch.cat([b_up, e4], dim=1))

        d4_up = nn.functional.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = self.up3(torch.cat([d4_up, e3], dim=1))

        d3_up = nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = self.up2(torch.cat([d3_up, e2], dim=1))

        d2_up = nn.functional.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.up1(torch.cat([d2_up, e1], dim=1))

        f = self.final_conv(d1)
        pooled = self.global_pool(f).squeeze(-1).squeeze(-1)  # (batch, 32)
        out = self.fc(pooled)  # (batch, 6)

        return out

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)

        # Log individual frequency band losses
        for i in range(self.hparams.num_frequencies):
            freq_loss = self.loss_fn(outputs[:, i : i + 1], targets[:, i : i + 1])
            self.log(f"train_loss_freq_{i}", freq_loss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)

        # Log individual frequency band losses
        for i in range(self.hparams.num_frequencies):
            freq_loss = self.loss_fn(outputs[:, i : i + 1], targets[:, i : i + 1])
            self.log(f"val_loss_freq_{i}", freq_loss, prog_bar=False)

        return loss

    def configure_optimizers(self):
        # Use AdamW which handles weight decay better
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-3
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
