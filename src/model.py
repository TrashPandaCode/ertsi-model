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

        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

        # Use a pretrained model from Places365 (captures spatial/room features better than ImageNet)
        base = load_model.resnet50_places365()

        # Extract features from the backbone (excluding classification layers)
        self.backbone = nn.Sequential(*list(base.children())[:-2])

        # Global pooling to handle different input image sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Frequency-aware feature extraction
        # This creates parallel pathways for each frequency band
        self.freq_specific_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(2048, 256, kernel_size=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                for _ in range(num_frequencies)
            ]
        )

        # Shared layers to extract common room acoustic features
        self.shared_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Final prediction heads - one for each frequency band
        self.prediction_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        256 + 256, 128
                    ),  # 256 from freq-specific + 256 from shared
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 1),
                )
                for _ in range(num_frequencies)
            ]
        )

        self.loss_fn = nn.MSELoss()

    def extract_features(self, x):
        """Extract shared features for visualization/analysis"""
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Compute shared features
        pooled_features = self.global_pool(features)
        shared_features = self.shared_layers(pooled_features)
        
        return shared_features

    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)

        # Compute shared features
        pooled_features = self.global_pool(features)
        shared_features = self.shared_layers(pooled_features)

        # Process through frequency-specific pathways
        freq_features = [
            freq_layer(features) for freq_layer in self.freq_specific_layers
        ]

        # Make predictions for each frequency
        outputs = []
        for i, pred_head in enumerate(self.prediction_heads):
            # Concatenate shared features with frequency-specific features
            combined = torch.cat([shared_features, freq_features[i]], dim=1)
            # Predict RT60 for this frequency band
            rt60 = pred_head(combined)
            outputs.append(rt60)

        # Stack predictions into a single tensor [batch_size, num_frequencies]
        return torch.cat(outputs, dim=1)

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
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
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