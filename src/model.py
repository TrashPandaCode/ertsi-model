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

        # Use ResNet50 - corrected from the comment
        base = load_model.resnet50_places365()

        # Extract features from the backbone (excluding classification layers)
        self.backbone = nn.Sequential(*list(base.children())[:-2])

        # Global pooling to handle different input image sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Correct feature dimensions for ResNet50
        backbone_features = 2048  # ResNet50 has 2048 features, not 512

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
                    nn.Conv2d(backbone_features, 256, kernel_size=1),  # Increased from 128
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                for _ in range(num_frequencies)
            ]
        )

        # Optimized shared layers - adjusted for ResNet50
        self.shared_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_features, 512),  # Increased from 256
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Slightly increased dropout for larger model
            nn.Linear(512, 256),  # Increased from 128
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # Simplified prediction heads - adjusted input size
        self.prediction_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        256 + 256, 128  # 256 from freq-specific + 256 from shared
                    ),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1),
                )
                for _ in range(num_frequencies)
            ]
        )

        # Use Huber loss for better robustness to outliers
        self.loss_fn = nn.HuberLoss(delta=0.1, reduction='none')
        
        # Frequency weights for weighted loss
        self.register_buffer('freq_weights', torch.tensor([1.0, 1.2, 1.5, 1.3, 1.1, 0.9]))

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
        
        # Compute frequency-weighted loss
        losses = self.loss_fn(outputs, targets)
        
        # Apply frequency weights (expand to match batch size)
        freq_weights = self.freq_weights[:outputs.size(1)].unsqueeze(0).expand_as(losses)
        weighted_losses = losses * freq_weights
        loss = weighted_losses.mean()
        
        # Calculate metrics
        mae = torch.abs(outputs - targets).mean()
        mse = torch.pow(outputs - targets, 2).mean()
        rmse = torch.sqrt(mse)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_rmse", rmse, on_step=False, on_epoch=True, prog_bar=False)
        
        # Log per-frequency metrics (less frequently to avoid clutter)
        if batch_idx % 50 == 0:
            for i in range(outputs.size(1)):
                freq_mae = torch.abs(outputs[:, i] - targets[:, i]).mean()
                freq_loss = losses[:, i].mean()
                self.log(f"train_mae_freq_{i}", freq_mae, on_step=False, on_epoch=True)
                self.log(f"train_loss_freq_{i}", freq_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        
        # Compute frequency-weighted loss
        losses = self.loss_fn(outputs, targets)
        freq_weights = self.freq_weights[:outputs.size(1)].unsqueeze(0).expand_as(losses)
        weighted_losses = losses * freq_weights
        loss = weighted_losses.mean()
        
        # Calculate metrics
        mae = torch.abs(outputs - targets).mean()
        mse = torch.pow(outputs - targets, 2).mean()
        rmse = torch.sqrt(mse)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True, prog_bar=False)
        
        # Log per-frequency validation metrics
        for i in range(outputs.size(1)):
            freq_mae = torch.abs(outputs[:, i] - targets[:, i]).mean()
            freq_loss = losses[:, i].mean()
            self.log(f"val_mae_freq_{i}", freq_mae, on_step=False, on_epoch=True)
            self.log(f"val_loss_freq_{i}", freq_loss, on_step=False, on_epoch=True)
        
        return {
            "val_loss": loss,
            "val_mae": mae,
            "val_rmse": rmse,
            "outputs": outputs.detach(),
            "targets": targets.detach()
        }

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