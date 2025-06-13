import torch
import torch.nn as nn
import pytorch_lightning as pl
from load_model import resnet50_places365


class RoomSpecificReverbCNN(pl.LightningModule):
    def __init__(self, num_frequencies=6, learning_rate=0.001, num_rooms=None):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_frequencies = num_frequencies
        self.num_rooms = num_rooms

        # Load pre-trained Places365 ResNet50 as backbone
        backbone_full = resnet50_places365()
        
        # Extract feature layers before global pooling
        # This gives us features with spatial dimensions (e.g., [batch, 2048, 7, 7])
        self.backbone = nn.Sequential(
            backbone_full.conv1,
            backbone_full.bn1,
            backbone_full.relu,
            backbone_full.maxpool,
            backbone_full.layer1,
            backbone_full.layer2,
            backbone_full.layer3,
            backbone_full.layer4
        )

        # Add spatial attention mechanism (receives 4D input)
        self.spatial_attention = SpatialAttentionModule(2048)

        # Add room embedding if room IDs are available
        if num_rooms:
            self.room_embedding = nn.Embedding(num_rooms, 256)
            feature_dim = 2048 + 256
        else:
            feature_dim = 2048

        # Enhanced regression head with room-aware features
        self.regression_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, num_frequencies),
        )

        # Multi-scale feature extraction
        self.multi_scale_features = MultiScaleFeatureExtractor()

    def forward(self, x, room_id=None):
        # Extract features from backbone (4D: [batch, 2048, 7, 7])
        features = self.backbone(x)

        # Apply spatial attention (reduces to 1D: [batch, 2048])
        features = self.spatial_attention(features)

        # Add multi-scale features
        multi_scale = self.multi_scale_features(x)
        features = features + multi_scale

        # Add room embedding if available
        if room_id is not None and self.num_rooms:
            room_emb = self.room_embedding(room_id)
            features = torch.cat([features, room_emb], dim=1)

        # Regression prediction
        rt60_pred = self.regression_head(features)
        return rt60_pred

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            images, targets, room_ids = batch
            predictions = self(images, room_ids)
        else:
            images, targets = batch
            predictions = self(images)

        # Standard MSE loss
        mse_loss = nn.functional.mse_loss(predictions, targets)

        # Room consistency loss (encourage similar predictions for same room)
        if len(batch) == 3 and len(torch.unique(room_ids)) > 1:
            room_loss = self.compute_room_consistency_loss(predictions, room_ids)
            total_loss = mse_loss + 0.1 * room_loss
            self.log("room_loss", room_loss, on_step=True, on_epoch=True)
        else:
            total_loss = mse_loss

        self.log("train_loss", total_loss, on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            images, targets, room_ids = batch
            predictions = self(images, room_ids)
        else:
            images, targets = batch
            predictions = self(images)

        # Calculate validation loss
        val_loss = nn.functional.mse_loss(predictions, targets)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate MAE for better interpretability
        mae = nn.functional.l1_loss(predictions, targets)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Add learning rate scheduler
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
            },
        }

    def compute_room_consistency_loss(self, predictions, room_ids):
        """Encourage consistent predictions within the same room"""
        room_loss = 0.0
        unique_rooms = torch.unique(room_ids)

        for room_id in unique_rooms:
            room_mask = room_ids == room_id
            room_predictions = predictions[room_mask]

            if len(room_predictions) > 1:
                # Compute variance within room predictions
                room_variance = torch.var(room_predictions, dim=0).mean()
                room_loss += room_variance

        return (
            room_loss / len(unique_rooms)
            if len(unique_rooms) > 0
            else torch.tensor(0.0, device=predictions.device)
        )


class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: [batch, channels, height, width]
        attention_weights = self.attention(x)
        # Apply attention and reduce spatial dimensions
        attended_features = x * attention_weights
        # Global average pooling to get [batch, channels]
        return torch.mean(attended_features, dim=[2, 3])


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                )
                for size in [(7, 7), (14, 14), (28, 28)]
            ]
        )
        self.fusion = nn.Linear(64 * 3, 2048)

    def forward(self, x):
        scale_features = []
        for scale_net in self.scales:
            scale_features.append(scale_net(x))

        combined = torch.cat(scale_features, dim=1)
        return self.fusion(combined)