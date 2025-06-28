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

    def __init__(self, num_frequencies=6, learning_rate=0.001, freeze_backbone=False, freeze_first_layers=False):
        super().__init__()
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters()
        
        # Use a pretrained model from Places365
        base = load_model.resnet50_places365()
        
        # Extract features from the backbone (excluding classification layers)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        
        # Freeze the backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone eingefroren")
        
        # Global pooling to handle different input image sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Frequency-aware feature extraction mit mehr Dropout
        self.freq_specific_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=1),
                nn.BatchNorm2d(256),  # BatchNorm hinzugefügt
                nn.ReLU(),
                nn.Dropout2d(0.3),  # 2D Dropout für Conv Layer
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            ) for _ in range(num_frequencies)
        ])
        
        # Freeze the first two layers of each frequency-specific pathway if specified
        if freeze_first_layers:
            for freq_layer in self.freq_specific_layers:
                # Only freeze the first two layers (Conv2d and BatchNorm)
                for i, layer in enumerate(freq_layer):
                    if i < 2:  # Conv2d und BatchNorm
                        for param in layer.parameters():
                            param.requires_grad = False
            print("Erste Schichten der frequenzspezifischen Layer eingefroren")
        
        # Shared layers with more regularization
        self.shared_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),  # BatchNorm 
            nn.ReLU(),
            nn.Dropout(0.5),  # higher Dropout
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # BatchNorm 
            nn.ReLU(),
            nn.Dropout(0.4)  # higher Dropout
        )
        
        # Final prediction heads mit mehr Regularisierung
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256 + 256, 128),
                nn.BatchNorm1d(128),  # BatchNorm hinzugefügt
                nn.ReLU(),
                nn.Dropout(0.4),  # Höherer Dropout
                nn.Linear(128, 64),  # Zusätzliche Schicht
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            ) for _ in range(num_frequencies)
        ])
        
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Compute shared features
        pooled_features = self.global_pool(features)
        shared_features = self.shared_layers(pooled_features)
        
        # Process through frequency-specific pathways
        freq_features = [freq_layer(features) for freq_layer in self.freq_specific_layers]
        
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
        
        # L2 Regularisierung hinzufügen
        l2_reg = torch.tensor(0., device=self.device)
        for param in self.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, p=2)
        
        # Stärkere L2 Regularisierung
        loss += 0.001 * l2_reg
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("l2_reg", l2_reg, prog_bar=False)
        
        # Log individual frequency band losses
        for i in range(self.hparams.num_frequencies):
            freq_loss = self.loss_fn(outputs[:, i:i+1], targets[:, i:i+1])
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
            freq_loss = self.loss_fn(outputs[:, i:i+1], targets[:, i:i+1])
            self.log(f"val_loss_freq_{i}", freq_loss, prog_bar=False)
            
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        total_norm = 0
        for param in self.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
    
        self.log("grad_norm", total_norm, prog_bar=True)
    
        if total_norm > 10.0:
            print(f"WARNING: Large gradient norm: {total_norm:.2f}")

    def configure_optimizers(self):
        # Unterschiedliche Lernraten für eingefrorene und nicht-eingefrorene Parameter
        if self.hparams.freeze_backbone or self.hparams.freeze_first_layers:
            # Niedrigere Lernrate für nicht-eingefrorene Parameter
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.hparams.learning_rate,
                weight_decay=5e-4  # Stärkere Weight Decay
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.hparams.learning_rate,
                weight_decay=5e-4
            )

    
        
        # Aggressiverer Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.3,  # Stärkere Reduktion
            patience=5,  # Weniger Geduld
            min_lr=1e-7  # Niedrigere minimale Lernrate
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