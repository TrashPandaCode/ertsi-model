import torch
from torch import nn
import pytorch_lightning as pl
import load_model
from torchvision import models
import torch.nn.functional as F


class ReverbCNN(pl.LightningModule):
    def __init__(self, num_frequencies=6, learning_rate=0.001, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained ResNet18 backbone (lighter for limited real data)
        self.backbone = load_model.resnet18_places365()
        
        # Remove the final classification layer but keep the avgpool
        # ResNet18 architecture: conv layers -> avgpool -> fc
        # We want to keep up to avgpool but remove fc
        self.backbone.fc = nn.Identity()
        
        # Get the number of features from the backbone after avgpool
        backbone_features = 512  # ResNet18 feature size after avgpool
        
        # Enhanced feature extraction layers with more regularization
        # Use LayerNorm instead of BatchNorm to avoid batch size issues
        self.feature_extractor = nn.Sequential(
            nn.Linear(backbone_features, 1024),  # Direct linear layer, no pooling needed
            nn.LayerNorm(1024),  # LayerNorm works with batch size 1
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Increased dropout for regularization
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),  # LayerNorm instead of BatchNorm1d
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # LayerNorm instead of BatchNorm1d
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # RT60 regression head
        self.rt60_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),  # LayerNorm instead of BatchNorm1d
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),  # LayerNorm instead of BatchNorm1d
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(64, num_frequencies)
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Freeze early layers of backbone for better generalization with limited data
        self._freeze_early_layers()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _freeze_early_layers(self):
        """Freeze early layers of the backbone for better generalization"""
        # Freeze first few layers
        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in ['conv1', 'bn1', 'layer1']):
                param.requires_grad = False
    
    def forward(self, x):
        # Extract features using the backbone (includes avgpool, outputs flattened features)
        features = self.backbone(x)
        
        # Ensure features are flattened (should already be from ResNet's avgpool + fc=Identity)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        # Process through feature extractor
        features = self.feature_extractor(features)
        
        # Predict RT60 values
        rt60_pred = self.rt60_head(features)
        
        return rt60_pred
    
    def training_step(self, batch, batch_idx):
        images, rt60_targets = batch
        rt60_pred = self(images)
        
        # Use Huber loss for robustness to outliers
        loss = F.huber_loss(rt60_pred, rt60_targets, delta=0.1)
        
        # Add L1 regularization for sparsity
        l1_lambda = 1e-5
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss = loss + l1_lambda * l1_norm
        
        # Calculate metrics
        mae = F.l1_loss(rt60_pred, rt60_targets)
        mse = F.mse_loss(rt60_pred, rt60_targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True)
        self.log('train_mse', mse, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, rt60_targets = batch
        rt60_pred = self(images)
        
        # Use same loss as training for consistency
        loss = F.huber_loss(rt60_pred, rt60_targets, delta=0.1)
        
        # Calculate metrics
        mae = F.l1_loss(rt60_pred, rt60_targets)
        mse = F.mse_loss(rt60_pred, rt60_targets)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mse', mse, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        # Use AdamW with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # More aggressive learning rate scheduling for real data
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.3,  # Reduce LR by 70%
            patience=5,   # Wait 5 epochs before reducing
            min_lr=1e-6,  # Minimum learning rate
            verbose=True,
            threshold=0.001,  # Minimum change threshold
            cooldown=2  # Wait 2 epochs after LR reduction
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
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        # Unfreeze more layers progressively
        current_epoch = self.current_epoch
        
        # Unfreeze layer2 after 20 epochs
        if current_epoch == 20:
            for name, param in self.backbone.named_parameters():
                if 'layer2' in name:
                    param.requires_grad = True
            print("Unfroze layer2 of backbone")
        
        # Unfreeze layer3 after 40 epochs
        elif current_epoch == 40:
            for name, param in self.backbone.named_parameters():
                if 'layer3' in name:
                    param.requires_grad = True
            print("Unfroze layer3 of backbone")
        
        # Unfreeze all layers after 60 epochs
        elif current_epoch == 60:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("Unfroze all backbone layers")


# Alternative lightweight model for very limited data
class LightweightReverbCNN(pl.LightningModule):
    def __init__(self, num_frequencies=6, learning_rate=0.001, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Lightweight CNN architecture
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Regression head with LayerNorm to avoid batch size issues
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),  # LayerNorm instead of BatchNorm1d
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),   # LayerNorm instead of BatchNorm1d
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_frequencies)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.regression_head(x)
        return x
    
    def training_step(self, batch, batch_idx):
        images, rt60_targets = batch
        rt60_pred = self(images)
        
        loss = F.huber_loss(rt60_pred, rt60_targets, delta=0.1)
        mae = F.l1_loss(rt60_pred, rt60_targets)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, rt60_targets = batch
        rt60_pred = self(images)
        
        loss = F.huber_loss(rt60_pred, rt60_targets, delta=0.1)
        mae = F.l1_loss(rt60_pred, rt60_targets)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }