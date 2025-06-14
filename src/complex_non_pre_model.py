import torch
from torch import nn
import pytorch_lightning as pl
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on acoustically important regions"""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention to emphasize important feature channels"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # Average pooling path
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # Max pooling path
        max_out = self.fc(self.max_pool(x).view(b, c))

        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention.expand_as(x)


class ResidualBlock(nn.Module):
    """Enhanced residual block with attention mechanisms"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply attention mechanisms
        out = self.channel_attention(out)
        out = self.spatial_attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales for better room understanding"""

    def __init__(self, in_channels):
        super().__init__()

        # Different kernel sizes to capture different spatial patterns
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=5, padding=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class FrequencyAwareHead(nn.Module):
    """Specialized head for frequency-dependent RT60 prediction"""

    def __init__(self, in_features, num_frequencies, hidden_dim=512):
        super().__init__()
        self.num_frequencies = num_frequencies

        # Shared feature processing
        self.shared_layers = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Frequency-specific branches
        self.freq_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim // 2, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                )
                for _ in range(num_frequencies)
            ]
        )

        # Cross-frequency interaction layer
        self.cross_freq_attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=8, batch_first=True
        )

        # Final prediction layer with frequency relationships
        self.final_layer = nn.Sequential(
            nn.Linear(num_frequencies, num_frequencies * 2),
            nn.ReLU(),
            nn.Linear(num_frequencies * 2, num_frequencies),
        )

    def forward(self, x):
        # Process through shared layers
        shared_features = self.shared_layers(x)

        # Get frequency-specific predictions
        freq_outputs = []
        intermediate_features = []

        for i, branch in enumerate(self.freq_branches):
            # Extract intermediate features for cross-frequency attention
            features = shared_features
            for layer in branch[:-1]:  # All layers except the last
                features = layer(features)
            intermediate_features.append(features)

            # Final prediction for this frequency
            freq_pred = branch[-1](features)
            freq_outputs.append(freq_pred)

        # Stack intermediate features for attention
        if len(intermediate_features) > 1:
            stacked_features = torch.stack(intermediate_features, dim=1)  # [B, F, D]
            attended_features, _ = self.cross_freq_attention(
                stacked_features, stacked_features, stacked_features
            )

            # Apply final predictions with cross-frequency information
            freq_outputs = torch.cat(freq_outputs, dim=1)  # [B, F]
            refined_output = self.final_layer(freq_outputs)
            return refined_output
        else:
            return torch.cat(freq_outputs, dim=1)


class AdvancedReverbCNN(pl.LightningModule):
    def __init__(
        self,
        num_frequencies=6,
        learning_rate=1e-4,
        weight_decay=1e-4,
        use_pretrained=True,
        freeze_backbone_epochs=2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Use EfficientNet-B3 as backbone for better efficiency
        if use_pretrained:
            self.backbone = efficientnet_b3(
                weights=EfficientNet_B3_Weights.IMAGENET1K_V1
            )
            # Remove the classifier
            self.backbone.classifier = nn.Identity()
            backbone_features = 1536  # EfficientNet-B3 output features
        else:
            self.backbone = efficientnet_b3(weights=None)
            self.backbone.classifier = nn.Identity()
            backbone_features = 1536

        # Multi-scale feature extraction (now works with 512 channels)
        self.multi_scale = MultiScaleFeatureExtractor(512)

        # Transition layer to reduce backbone features
        self.feature_transition = nn.Sequential(
            nn.Conv2d(backbone_features, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Additional residual blocks for room-specific features
        self.room_features = nn.Sequential(
            ResidualBlock(
                512,
                512,
                stride=1,
                downsample=nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=1, bias=False), nn.BatchNorm2d(512)
                ),
            ),
            ResidualBlock(512, 512, stride=1),
            ResidualBlock(
                512,
                256,
                stride=1,
                downsample=nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256)
                ),
            ),
        )

        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # Frequency-aware prediction head
        self.prediction_head = FrequencyAwareHead(
            in_features=256 * 2,  # Concat of avg and max pooling
            num_frequencies=num_frequencies,
            hidden_dim=1024,
        )

        # Loss function with frequency-aware weighting
        self.register_buffer("freq_weights", torch.ones(num_frequencies))
        self.loss_fn = nn.SmoothL1Loss()  # More robust to outliers than MSE

        # For learning rate scheduling
        self.freeze_backbone_epochs = freeze_backbone_epochs

    def forward(self, x):
        # Extract backbone features
        features = self.backbone.features(x)

        # Reduce feature dimensions
        features = self.feature_transition(features)

        # Multi-scale feature extraction
        multi_scale_features = self.multi_scale(features)

        # Room-specific feature processing
        room_features = self.room_features(multi_scale_features)

        # Global pooling
        avg_pool = self.global_pool(room_features).flatten(1)
        max_pool = self.global_max_pool(room_features).flatten(1)

        # Combine pooled features
        combined_features = torch.cat([avg_pool, max_pool], dim=1)

        # Frequency-aware prediction
        rt60_pred = self.prediction_head(combined_features)

        return rt60_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Frequency-weighted loss
        loss = self.compute_weighted_loss(y_hat, y)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)

        # Log per-frequency metrics
        with torch.no_grad():
            mae = torch.abs(y_hat - y).mean(dim=0)
            for i, freq_mae in enumerate(mae):
                self.log(f"train_mae_freq_{i}", freq_mae)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.compute_weighted_loss(y_hat, y)
        mae = torch.abs(y_hat - y).mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)

        # Log per-frequency validation metrics
        freq_mae = torch.abs(y_hat - y).mean(dim=0)
        for i, f_mae in enumerate(freq_mae):
            self.log(f"val_mae_freq_{i}", f_mae)

        return {"val_loss": loss, "val_mae": mae}

    def compute_weighted_loss(self, y_hat, y):
        """Compute frequency-weighted loss"""
        per_freq_loss = torch.abs(y_hat - y)  # L1 loss per frequency
        weighted_loss = (per_freq_loss * self.freq_weights).mean()
        return weighted_loss

    def configure_optimizers(self):
        # Different learning rates for backbone and head
        backbone_params = list(self.backbone.parameters())
        head_params = (
            list(self.multi_scale.parameters())
            + list(self.room_features.parameters())
            + list(self.prediction_head.parameters())
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.learning_rate * 0.1},
                {"params": head_params, "lr": self.hparams.learning_rate},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def on_train_epoch_start(self):
        # Freeze backbone for first few epochs if using pretrained weights
        if self.current_epoch < self.freeze_backbone_epochs:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def update_frequency_weights(self, train_loader):
        """Update frequency weights based on training data statistics"""
        rt60_values = []
        for batch in train_loader:
            _, y = batch
            rt60_values.append(y)

        all_rt60 = torch.cat(rt60_values, dim=0)

        # Weight inversely proportional to variance (more weight to harder frequencies)
        freq_var = torch.var(all_rt60, dim=0)
        weights = 1.0 / (freq_var + 1e-8)
        weights = weights / weights.sum() * len(weights)  # Normalize

        self.freq_weights.data = weights.to(self.device)
