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

    def __init__(self, num_frequencies=6, learning_rate=0.001, total_steps=None):
        super().__init__()

        # Store total_steps for scheduler
        self.total_steps = total_steps

        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

        # Use a more efficient backbone - try ResNet18 first
        base = load_model.resnet18_places365()  # Lighter than ResNet50

        # Extract features from the backbone (excluding classification layers)
        self.backbone = nn.Sequential(*list(base.children())[:-2])

        # Global pooling to handle different input image sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))


        # Reduced feature dimensions for efficiency
        backbone_features = 512  # ResNet18 has 512 features vs 2048 for ResNet50


        # More efficient frequency-specific layers with depthwise separable convolutions
        self.freq_specific_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        backbone_features,
                        backbone_features,
                        kernel_size=3,
                        groups=backbone_features,
                        padding=1,
                    ),
                    nn.Conv2d(backbone_features, 128, kernel_size=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                for _ in range(num_frequencies)
            ]
        )

        # Optimized shared layers
        self.shared_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Reduced dropout
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Simplified prediction heads
        self.prediction_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        128 + 128, 64
                    ),  # 128 from freq-specific + 128 from shared
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1),
                )
                for _ in range(num_frequencies)
            ]
        )

        # Use Huber loss for better robustness to outliers
        self.loss_fn = nn.HuberLoss(delta=0.1)

    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)

        # Compute shared features
        pooled_features = self.global_pool(features)
        shared_features = self.shared_layers(pooled_features)

        # Process through frequency-specific pathways efficiently
        freq_features = []
        for freq_layer in self.freq_specific_layers:
            freq_feat = freq_layer(features)
            freq_features.append(freq_feat)

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
        # Use AdamW with improved parameters
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-3,  # Increased weight decay
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        if self.total_steps is not None:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate * 10,
                total_steps=self.total_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        else:
            # Fallback to a simpler scheduler if total_steps not provided
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
