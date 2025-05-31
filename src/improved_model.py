import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Optional, Tuple, List
import load_model


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on acoustically relevant regions"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels // 8, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = F.relu(attention)
        attention = self.conv3(attention)
        attention = self.sigmoid(attention)
        return x * attention


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales for better room understanding"""

    def __init__(self, backbone):
        super().__init__()
        # Extract intermediate layers from ResNet
        self.layer0 = nn.Sequential(
            *list(backbone.children())[:4]
        )  # Conv + BN + ReLU + MaxPool
        self.layer1 = backbone.layer1  # 64 -> 256
        self.layer2 = backbone.layer2  # 256 -> 512
        self.layer3 = backbone.layer3  # 512 -> 1024
        self.layer4 = backbone.layer4  # 1024 -> 2048

        # Spatial attention for each scale
        self.attention1 = SpatialAttention(256)
        self.attention2 = SpatialAttention(512)
        self.attention3 = SpatialAttention(1024)
        self.attention4 = SpatialAttention(2048)

        # Feature fusion layers
        self.fusion_conv1 = nn.Conv2d(256, 128, 1)
        self.fusion_conv2 = nn.Conv2d(512, 128, 1)
        self.fusion_conv3 = nn.Conv2d(1024, 128, 1)
        self.fusion_conv4 = nn.Conv2d(2048, 128, 1)

        self.final_fusion = nn.Conv2d(512, 256, 3, padding=1)

    def forward(self, x):
        # Extract multi-scale features
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Apply spatial attention
        x1_att = self.attention1(x1)
        x2_att = self.attention2(x2)
        x3_att = self.attention3(x3)
        x4_att = self.attention4(x4)

        # Reduce channels and upsample to common size
        target_size = x4_att.shape[2:]

        f1 = F.interpolate(
            self.fusion_conv1(x1_att), target_size, mode="bilinear", align_corners=False
        )
        f2 = F.interpolate(
            self.fusion_conv2(x2_att), target_size, mode="bilinear", align_corners=False
        )
        f3 = F.interpolate(
            self.fusion_conv3(x3_att), target_size, mode="bilinear", align_corners=False
        )
        f4 = self.fusion_conv4(x4_att)

        # Concatenate and fuse
        fused = torch.cat([f1, f2, f3, f4], dim=1)
        fused = self.final_fusion(fused)

        return fused, x4_att  # Return both fused features and final layer


class FrequencyInteractionModule(nn.Module):
    """Model interactions between different frequency bands"""

    def __init__(self, feature_dim: int, num_frequencies: int):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.feature_dim = feature_dim

        # Cross-frequency attention
        self.cross_freq_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # Frequency position encoding
        self.freq_pos_encoding = nn.Parameter(
            torch.randn(num_frequencies, feature_dim) * 0.02
        )

        # Frequency-specific transformations
        self.freq_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(feature_dim // 2, feature_dim),
                )
                for _ in range(num_frequencies)
            ]
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, shared_features):
        batch_size = shared_features.shape[0]

        # Create frequency-specific features
        freq_features = []
        for i, transform in enumerate(self.freq_transforms):
            freq_feat = transform(shared_features) + self.freq_pos_encoding[i]
            freq_features.append(freq_feat)

        # Stack for attention
        freq_stack = torch.stack(freq_features, dim=1)  # [B, F, D]

        # Apply cross-frequency attention
        attended_features, attention_weights = self.cross_freq_attention(
            freq_stack, freq_stack, freq_stack
        )

        # Residual connection and layer norm
        attended_features = self.layer_norm(attended_features + freq_stack)

        return attended_features, attention_weights


class PerceptualLoss(nn.Module):
    """Perceptually-informed loss for RT60 prediction"""

    def __init__(
        self,
        frequencies: List[int],
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
    ):
        super().__init__()
        self.frequencies = torch.tensor(frequencies, dtype=torch.float32)
        self.alpha = alpha  # MSE weight
        self.beta = beta  # Frequency smoothness weight
        self.gamma = gamma  # Perceptual weighting

        # Perceptual importance weights (more weight on speech frequencies)
        freq_weights = []
        for freq in frequencies:
            if 250 <= freq <= 4000:  # Speech range
                freq_weights.append(1.2)
            elif 125 <= freq <= 8000:  # Extended speech range
                freq_weights.append(1.0)
            else:
                freq_weights.append(0.8)

        self.register_buffer("perceptual_weights", torch.tensor(freq_weights))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Base MSE loss with perceptual weighting
        mse_per_freq = (pred - target) ** 2
        weighted_mse = (mse_per_freq * self.perceptual_weights).mean()

        # Frequency smoothness loss (adjacent frequencies should be similar)
        pred_diff = torch.diff(pred, dim=1)
        target_diff = torch.diff(target, dim=1)
        smoothness_loss = F.mse_loss(pred_diff, target_diff)

        # Total variation loss (penalize unrealistic RT60 curves)
        tv_loss = torch.mean(torch.abs(pred_diff))

        total_loss = (
            self.alpha * weighted_mse
            + self.beta * smoothness_loss
            + self.gamma * tv_loss
        )

        return total_loss


class ImprovedReverbCNN(pl.LightningModule):
    """Improved RT60 prediction model with advanced architecture"""

    def __init__(
        self,
        num_frequencies: int = 6,
        learning_rate: float = 0.001,
        frequencies: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        use_scheduler: bool = True,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_frequencies = num_frequencies
        self.frequencies = frequencies or [250, 500, 1000, 2000, 4000, 8000]
        self.dropout_rate = dropout_rate
        self.use_scheduler = use_scheduler
        self.warmup_epochs = warmup_epochs
        self.learning_rate = learning_rate

        # Load pretrained backbone
        backbone = load_model.resnet50_places365()

        # Multi-scale feature extractor
        self.feature_extractor = MultiScaleFeatureExtractor(backbone)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Shared feature processing
        self.shared_processor = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
        )

        # Frequency interaction module
        self.freq_interaction = FrequencyInteractionModule(256, num_frequencies)

        # Final prediction heads with residual connections
        self.prediction_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Dropout(dropout_rate / 2),
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Linear(64, 1),
                )
                for _ in range(num_frequencies)
            ]
        )

        # Uncertainty estimation layers (for Monte Carlo Dropout)
        self.uncertainty_dropout = nn.Dropout(0.1)

        # Loss function
        self.loss_fn = PerceptualLoss(self.frequencies)

        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def enable_dropout_for_uncertainty(self):
        """Enable dropout layers for uncertainty estimation"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Extract multi-scale features
        fused_features, final_features = self.feature_extractor(x)

        # Global pooling and shared processing
        pooled = self.global_pool(fused_features).flatten(1)
        shared_features = self.shared_processor(pooled)

        # Frequency interaction modeling
        freq_features, attention_weights = self.freq_interaction(shared_features)

        # Make predictions for each frequency
        predictions = []
        for i, pred_head in enumerate(self.prediction_heads):
            freq_specific_features = freq_features[:, i, :]

            # Apply uncertainty dropout during inference if enabled
            if not self.training:
                freq_specific_features = self.uncertainty_dropout(
                    freq_specific_features
                )

            pred = pred_head(freq_specific_features)
            predictions.append(pred)

        # Stack predictions
        output = torch.cat(predictions, dim=1)

        return output

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        loss = self.loss_fn(outputs, targets)

        # Calculate metrics
        mse = F.mse_loss(outputs, targets)
        mae = F.l1_loss(outputs, targets)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_mse", mse, on_step=False, on_epoch=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True)

        # Log frequency-specific losses
        for i in range(self.num_frequencies):
            freq_loss = F.mse_loss(outputs[:, i], targets[:, i])
            self.log(
                f"train_freq_{self.frequencies[i]}Hz",
                freq_loss,
                on_step=False,
                on_epoch=True,
            )

        self.training_step_outputs.append(
            {"loss": loss, "outputs": outputs.detach(), "targets": targets.detach()}
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)

        loss = self.loss_fn(outputs, targets)

        # Calculate metrics
        mse = F.mse_loss(outputs, targets)
        mae = F.l1_loss(outputs, targets)

        # Calculate R²
        ss_res = torch.sum((targets - outputs) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - ss_res / ss_tot

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mse", mse, on_step=False, on_epoch=True)
        self.log("val_mae", mae, on_step=False, on_epoch=True)
        self.log("val_r2", r2, on_step=False, on_epoch=True)

        # Log frequency-specific metrics
        for i in range(self.num_frequencies):
            freq_mse = F.mse_loss(outputs[:, i], targets[:, i])
            freq_mae = F.l1_loss(outputs[:, i], targets[:, i])

            # R² for this frequency
            freq_ss_res = torch.sum((targets[:, i] - outputs[:, i]) ** 2)
            freq_ss_tot = torch.sum((targets[:, i] - torch.mean(targets[:, i])) ** 2)
            freq_r2 = 1 - freq_ss_res / freq_ss_tot

            self.log(
                f"val_freq_{self.frequencies[i]}Hz_mse",
                freq_mse,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"val_freq_{self.frequencies[i]}Hz_r2",
                freq_r2,
                on_step=False,
                on_epoch=True,
            )

        self.validation_step_outputs.append(
            {"loss": loss, "outputs": outputs.detach(), "targets": targets.detach()}
        )

        return loss

    def on_train_epoch_end(self):
        if self.training_step_outputs:
            all_outputs = torch.cat([x["outputs"] for x in self.training_step_outputs])
            all_targets = torch.cat([x["targets"] for x in self.training_step_outputs])

            # Calculate epoch-level correlation between frequencies
            if all_outputs.shape[0] > 1:
                pred_corr = torch.corrcoef(all_outputs.T)
                target_corr = torch.corrcoef(all_targets.T)
                corr_diff = torch.norm(pred_corr - target_corr, p="fro")
                self.log("train_corr_diff", corr_diff, on_epoch=True)

            self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            all_outputs = torch.cat(
                [x["outputs"] for x in self.validation_step_outputs]
            )
            all_targets = torch.cat(
                [x["targets"] for x in self.validation_step_outputs]
            )

            # Calculate epoch-level correlation between frequencies
            if all_outputs.shape[0] > 1:
                pred_corr = torch.corrcoef(all_outputs.T)
                target_corr = torch.corrcoef(all_targets.T)
                corr_diff = torch.norm(pred_corr - target_corr, p="fro")
                self.log("val_corr_diff", corr_diff, on_epoch=True)

            self.validation_step_outputs.clear()

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with Monte Carlo Dropout uncertainty estimation"""
        self.eval()
        self.enable_dropout_for_uncertainty()

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        return mean_pred, std_pred

    def configure_optimizers(self):
        # Use AdamW with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        if not self.use_scheduler:
            return optimizer

        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=self.warmup_epochs
        )

        # Combine warmup and cosine annealing
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[self.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# Example usage and training modifications
class ProgressiveTrainer:
    """Implement progressive training strategy"""

    def __init__(self, model, train_dataset, val_dataset, frequencies):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.frequencies = frequencies

    def create_difficulty_stages(self, num_stages: int = 3):
        """Create training stages based on RT60 variance"""
        # Calculate variance for each sample
        variances = []
        for _, rt60 in self.train_dataset.entries:
            variance = torch.var(rt60).item()
            variances.append(variance)

        # Sort indices by variance (easy to hard)
        sorted_indices = np.argsort(variances)

        # Create stages
        stage_datasets = []
        samples_per_stage = len(self.train_dataset) // num_stages

        for i in range(num_stages):
            end_idx = (i + 1) * samples_per_stage
            if i == num_stages - 1:  # Last stage gets all remaining samples
                end_idx = len(sorted_indices)

            stage_indices = sorted_indices[:end_idx]
            stage_datasets.append(
                torch.utils.data.Subset(self.train_dataset, stage_indices)
            )

        return stage_datasets

    def train_progressive(self, trainer, epochs_per_stage: List[int]):
        """Train progressively through difficulty stages"""
        stage_datasets = self.create_difficulty_stages()

        for stage_idx, (stage_data, epochs) in enumerate(
            zip(stage_datasets, epochs_per_stage)
        ):
            print(
                f"Training stage {stage_idx + 1}/{len(stage_datasets)} with {len(stage_data)} samples"
            )

            stage_loader = torch.utils.data.DataLoader(
                stage_data, batch_size=32, shuffle=True, num_workers=4
            )

            val_loader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=32, shuffle=False, num_workers=4
            )

            trainer.fit(self.model, stage_loader, val_loader)


# Ensemble Model for improved performance
class EnsembleReverbCNN(nn.Module):
    """Ensemble of ReverbCNN models for improved accuracy"""

    def __init__(self, models: List[ImprovedReverbCNN]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and std of ensemble predictions"""
        predictions = []

        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        return mean_pred, std_pred

    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with both ensemble and MC dropout uncertainty"""
        ensemble_preds = []
        ensemble_stds = []

        for model in self.models:
            mean_pred, std_pred = model.predict_with_uncertainty(x, n_samples=10)
            ensemble_preds.append(mean_pred)
            ensemble_stds.append(std_pred)

        # Combine ensemble predictions
        ensemble_preds = torch.stack(ensemble_preds)
        ensemble_mean = ensemble_preds.mean(dim=0)

        # Combine uncertainties (both epistemic and aleatoric)
        ensemble_stds = torch.stack(ensemble_stds)
        aleatoric_uncertainty = ensemble_stds.mean(
            dim=0
        )  # Average MC dropout uncertainty
        epistemic_uncertainty = ensemble_preds.std(dim=0)  # Ensemble disagreement

        total_uncertainty = torch.sqrt(
            aleatoric_uncertainty**2 + epistemic_uncertainty**2
        )

        return ensemble_mean, total_uncertainty
