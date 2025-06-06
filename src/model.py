import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, List
from backbone_models import get_backbone


class FrequencyInteractionModule(nn.Module):
    """Module for modeling interactions between different frequency bands"""
    
    def __init__(self, feature_dim: int, num_frequencies: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_frequencies = num_frequencies
        
        # Cross-frequency attention
        self.freq_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Frequency-specific transformations
        self.freq_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, feature_dim),
            )
            for _ in range(num_frequencies)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, shared_features):
        # shared_features: [batch_size, feature_dim]
        batch_size = shared_features.shape[0]
        
        # Expand features for each frequency
        freq_features = shared_features.unsqueeze(1).expand(-1, self.num_frequencies, -1)
        # freq_features: [batch_size, num_frequencies, feature_dim]
        
        # Apply frequency-specific transformations
        transformed_features = []
        for i, transform in enumerate(self.freq_transforms):
            freq_specific = transform(freq_features[:, i, :])
            transformed_features.append(freq_specific.unsqueeze(1))
        
        transformed_features = torch.cat(transformed_features, dim=1)
        # transformed_features: [batch_size, num_frequencies, feature_dim]
        
        # Apply cross-frequency attention
        attended_features, attention_weights = self.freq_attention(
            transformed_features, transformed_features, transformed_features
        )
        
        # Apply layer normalization and residual connection
        output = self.layer_norm(attended_features + transformed_features)
        
        return output, attention_weights


class PerceptualLoss(nn.Module):
    """Perceptual loss for RT60 prediction with frequency-aware weighting"""
    
    def __init__(self, frequencies: List[int]):
        super().__init__()
        self.frequencies = frequencies
        
        # Frequency-dependent weights (lower frequencies often more important for perception)
        weights = []
        for freq in frequencies:
            if freq <= 500:
                weights.append(1.5)  # Higher weight for low frequencies
            elif freq <= 2000:
                weights.append(1.0)  # Normal weight for mid frequencies
            else:
                weights.append(0.8)  # Lower weight for high frequencies
        
        self.register_buffer('freq_weights', torch.tensor(weights))
        
    def forward(self, predictions, targets):
        # MSE loss with frequency weighting
        mse_per_freq = F.mse_loss(predictions, targets, reduction='none')
        weighted_mse = mse_per_freq * self.freq_weights.unsqueeze(0)
        
        # Add smoothness penalty (encourages similar RT60 values for adjacent frequencies)
        smoothness_loss = 0
        for i in range(len(self.frequencies) - 1):
            freq_diff = torch.abs(predictions[:, i] - predictions[:, i + 1])
            target_diff = torch.abs(targets[:, i] - targets[:, i + 1])
            smoothness_loss += F.mse_loss(freq_diff, target_diff)
        
        total_loss = weighted_mse.mean() + 0.1 * smoothness_loss
        return total_loss


class AdaptiveMultiScaleFeatureExtractor(nn.Module):
    """Adaptive feature extractor that works with different backbone architectures"""
    
    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone = get_backbone(backbone_name)
        self.backbone_name = backbone_name
        
        # Adaptive fusion layers based on backbone output
        self.feature_dim = self.backbone.feature_dim
        
        # Create adaptive pooling and fusion
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((16, 16)) for _ in range(4)
        ])
        
        # Feature dimension adaptation
        self.feature_adapters = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
        
        # Common target dimension
        target_dim = 128
        
        for i in range(4):
            # These will be set based on actual backbone output
            self.feature_adapters.append(nn.Identity())
            self.attention_modules.append(self._create_attention_module(target_dim))
        
        self.final_fusion = nn.Conv2d(target_dim * 4, 256, 3, padding=1)
        self._initialized = False
    
    def _create_attention_module(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
    
    def _initialize_adapters(self, features):
        """Initialize adapters based on actual feature dimensions"""
        if self._initialized:
            return
        
        target_dim = 128
        for i, feat in enumerate(features):
            input_dim = feat.shape[1]
            self.feature_adapters[i] = nn.Conv2d(input_dim, target_dim, 1).to(feat.device)
        
        self._initialized = True
    
    def forward(self, x):
        # Get multi-scale features from backbone
        features = self.backbone(x)
        
        # Initialize adapters if needed
        if not self._initialized:
            self._initialize_adapters(features)
        
        # Process each scale
        adapted_features = []
        target_size = (16, 16)
        
        for i, feat in enumerate(features):
            # Adapt feature dimensions
            adapted = self.feature_adapters[i](feat)
            
            # Resize to common spatial size
            if adapted.shape[2:] != target_size:
                adapted = F.interpolate(adapted, size=target_size, mode='bilinear', align_corners=False)
            
            # Apply attention
            attention = self.attention_modules[i](adapted)
            adapted = adapted * attention
            
            adapted_features.append(adapted)
        
        # Fuse all scales
        fused = torch.cat(adapted_features, dim=1)
        output = self.final_fusion(fused)
        
        return output, features[-1]  # Return fused features and final layer


class ReverbCNNComparison(pl.LightningModule):
    """Modified ReverbCNN for backbone comparison"""
    
    def __init__(
        self,
        backbone_name: str = "resnet50_places365",
        num_frequencies: int = 6,
        learning_rate: float = 0.001,
        frequencies: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
        use_scheduler: bool = True,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.backbone_name = backbone_name
        self.num_frequencies = num_frequencies
        self.frequencies = frequencies or [250, 500, 1000, 2000, 4000, 8000]
        self.dropout_rate = dropout_rate
        self.use_scheduler = use_scheduler
        self.warmup_epochs = warmup_epochs
        self.learning_rate = learning_rate
        
        # Adaptive feature extractor
        self.feature_extractor = AdaptiveMultiScaleFeatureExtractor(backbone_name)
        
        # Rest of the architecture remains the same
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
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
        
        # Prediction heads
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            for _ in range(num_frequencies)
        ])
        
        # Loss function
        self.loss_fn = PerceptualLoss(self.frequencies)
        
        # Metrics storage
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        fused_features, final_features = self.feature_extractor(x)
        
        # Global pooling and shared processing
        pooled = self.global_pool(fused_features).flatten(1)
        shared_features = self.shared_processor(pooled)
        
        # Frequency interaction modeling
        freq_features, attention_weights = self.freq_interaction(shared_features)
        
        # Make predictions
        predictions = []
        for i, pred_head in enumerate(self.prediction_heads):
            freq_specific_features = freq_features[:, i, :]
            pred = pred_head(freq_specific_features)
            predictions.append(pred)
        
        output = torch.cat(predictions, dim=1)
        return output
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        
        loss = self.loss_fn(outputs, targets)
        
        # Calculate metrics
        mse = F.mse_loss(outputs, targets)
        mae = F.l1_loss(outputs, targets)
        
        # Log metrics with backbone name
        self.log(f"train_loss_{self.backbone_name}", loss, prog_bar=True)
        self.log(f"train_mse_{self.backbone_name}", mse)
        self.log(f"train_mae_{self.backbone_name}", mae)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        
        loss = self.loss_fn(outputs, targets)
        mse = F.mse_loss(outputs, targets)
        mae = F.l1_loss(outputs, targets)
        
        # Calculate R²
        ss_res = torch.sum((targets - outputs) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        # Log metrics with backbone name
        self.log(f"val_loss_{self.backbone_name}", loss, prog_bar=True)
        self.log(f"val_mse_{self.backbone_name}", mse)
        self.log(f"val_mae_{self.backbone_name}", mae)
        self.log(f"val_r2_{self.backbone_name}", r2)
        
        return {"val_loss": loss, "val_mae": mae, "val_r2": r2}
    
    def configure_optimizers(self):
        # Use different learning rates for different backbones if needed
        lr_multipliers = {
            "resnet50_places365": 1.0,
            "dinov2": 0.1,  # Lower LR for pretrained transformers
            "clip": 0.1,  # OpenCLIP ViT-B-32
            "clip_vit_l": 0.05,  # Even lower for larger models
            "clip_convnext": 0.1,
            "efficientnet_b4": 1.0,
            "sam_b": 0.1,
        }
        
        adjusted_lr = self.learning_rate * lr_multipliers.get(self.backbone_name, 1.0)
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=adjusted_lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )
        
        if not self.use_scheduler:
            return optimizer
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }