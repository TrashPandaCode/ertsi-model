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

    def __init__(
        self, num_frequencies=6, learning_rate=0.001, backbone="resnet50_places365"
    ):
        super().__init__()

        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

        # Get backbone and feature dimensions
        self.backbone, self.feature_dim, self.needs_pool = load_model.get_model_info(
            backbone
        )

        # Global pooling to handle different input image sizes (if needed)
        if self.needs_pool:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.global_pool = None

        # Frequency-aware feature extraction
        # This creates parallel pathways for each frequency band
        self.freq_specific_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.feature_dim, 256, kernel_size=1),
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
            nn.Linear(self.feature_dim, 512),
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

    def forward(self, x):
        # Ensure input is on the same device as the model
        if x.device != self.device:
            x = x.to(self.device)

        # Extract features using the backbone
        features = self.backbone(x)

        # Handle different backbone output formats
        if isinstance(features, tuple):
            features = features[0]  # Some models return tuples

        # Ensure features are on the correct device
        if features.device != self.device:
            features = features.to(self.device)

        # Apply global pooling if needed
        if self.global_pool is not None:
            pooled_features = self.global_pool(features)
        else:
            # For models that already return pooled features
            if len(features.shape) == 4:  # Still has spatial dimensions
                pooled_features = nn.AdaptiveAvgPool2d((1, 1))(features)
            else:
                pooled_features = features.unsqueeze(-1).unsqueeze(
                    -1
                )  # Add spatial dims for compatibility

        # Compute shared features
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

        # Ensure inputs and targets are on the correct device
        if inputs.device != self.device:
            inputs = inputs.to(self.device)
        if targets.device != self.device:
            targets = targets.to(self.device)

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

        # Ensure inputs and targets are on the correct device
        if inputs.device != self.device:
            inputs = inputs.to(self.device)
        if targets.device != self.device:
            targets = targets.to(self.device)

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

    def on_train_start(self):
        """Ensure all components are on the correct device when training starts"""
        # Move backbone to device if it's a lambda function
        if hasattr(self.backbone, "to"):
            self.backbone = self.backbone.to(self.device)

    def on_validation_start(self):
        """Ensure all components are on the correct device when validation starts"""
        # Move backbone to device if it's a lambda function
        if hasattr(self.backbone, "to"):
            self.backbone = self.backbone.to(self.device)
