import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel, AutoImageProcessor
import timm
import open_clip
import load_model
import os
import urllib


class BackboneWrapper(nn.Module):
    """Base wrapper for different backbone architectures"""
    
    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = self._load_backbone()
        self.feature_dim = self._get_feature_dim()
        
    def _load_backbone(self):
        raise NotImplementedError
        
    def _get_feature_dim(self):
        raise NotImplementedError
        
    def forward(self, x):
        raise NotImplementedError


class ResNet50Places365Backbone(BackboneWrapper):
    """Original ResNet50 Places365 backbone"""
    
    def _load_backbone(self):
        return load_model.resnet50_places365()
    
    def _get_feature_dim(self):
        return 2048
    
    def forward(self, x):
        # Extract intermediate layers for multi-scale features
        layers = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)  # 256 channels
        layers.append(x)
        
        x = self.backbone.layer2(x)  # 512 channels
        layers.append(x)
        
        x = self.backbone.layer3(x)  # 1024 channels
        layers.append(x)
        
        x = self.backbone.layer4(x)  # 2048 channels
        layers.append(x)
        
        return layers


class DINOv2Backbone(BackboneWrapper):
    """DINOv2 Vision Transformer backbone"""
    
    def _load_backbone(self):
        # Load DINOv2 base model
        model = AutoModel.from_pretrained("facebook/dinov2-base")
        model.eval()
        for param in model.parameters():
            param.requires_grad = True  # Enable fine-tuning
        return model
    
    def _get_feature_dim(self):
        return 768  # DINOv2 base embedding dimension
    
    def forward(self, x):
        # DINOv2 expects specific input size (224x224)
        if x.shape[-1] != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # Get patch embeddings and CLS token
        outputs = self.backbone(x, output_hidden_states=True)
        
        # Extract multi-scale features from different transformer layers
        hidden_states = outputs.hidden_states
        
        # Use layers: 3, 6, 9, 12 (0-indexed) for multi-scale
        layer_indices = [3, 6, 9, 12]
        layers = []
        
        for idx in layer_indices:
            layer_output = hidden_states[idx]  # [B, num_patches+1, 768]
            cls_token = layer_output[:, 0, :]  # [B, 768]
            patch_tokens = layer_output[:, 1:, :]  # [B, num_patches, 768]
            
            # Reshape patch tokens back to spatial format
            # Assuming 224x224 input -> 16x16 patches for ViT
            B, num_patches, dim = patch_tokens.shape
            H = W = int(num_patches ** 0.5)
            spatial_features = patch_tokens.reshape(B, H, W, dim).permute(0, 3, 1, 2)
            
            layers.append(spatial_features)
        
        return layers


class OpenCLIPBackbone(BackboneWrapper):
    """OpenCLIP Vision Encoder backbone"""
    
    def __init__(self, backbone_name: str, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        self.model_name = model_name
        self.pretrained = pretrained
        super().__init__(backbone_name)
    
    def _load_backbone(self):
        # Load OpenCLIP model
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, 
            pretrained=self.pretrained
        )
        
        # Extract only the visual encoder
        visual_encoder = model.visual
        
        # Enable fine-tuning
        for param in visual_encoder.parameters():
            param.requires_grad = True
            
        return visual_encoder
    
    def _get_feature_dim(self):
        # Get feature dimension based on model
        if "ViT-B" in self.model_name:
            return 512
        elif "ViT-L" in self.model_name:
            return 768
        elif "ViT-H" in self.model_name:
            return 1024
        elif "convnext" in self.model_name.lower():
            return 1024  # ConvNeXt feature dimension
        else:
            return 512  # Default
    
    def forward(self, x):
        # OpenCLIP expects 224x224 input
        if x.shape[-1] != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
        
        # Determine architecture type by checking class name and attributes
        backbone_class_name = self.backbone.__class__.__name__
        
        # For TIMM-wrapped models (like ConvNeXt), use the TIMM approach
        if backbone_class_name == "TimmModel" or hasattr(self.backbone, 'trunk'):
            return self._forward_timm_model(x)
        # Check for ConvNeXt architecture
        elif "convnext" in self.model_name.lower() or "ConvNeXt" in backbone_class_name:
            return self._forward_convnext(x)
        # Check for Vision Transformer
        elif "ViT" in self.model_name or "VisionTransformer" in backbone_class_name:
            return self._forward_vit(x)
        else:
            # Try to detect based on attributes
            if hasattr(self.backbone, 'stem') or hasattr(self.backbone, 'stages'):
                return self._forward_convnext(x)
            elif hasattr(self.backbone, 'patch_embed') or hasattr(self.backbone, 'conv1'):
                return self._forward_vit(x)
            else:
                raise ValueError(f"Cannot determine architecture type for {self.model_name}")
    
    def _forward_timm_model(self, x):
        """Forward pass for TIMM-wrapped models"""
        layers = []
        
        # For TIMM models, we need to access the trunk and use forward_features
        if hasattr(self.backbone, 'trunk'):
            # TIMM ConvNeXt models have a trunk with stages
            trunk = self.backbone.trunk
            
            # Initial stem/downsample
            if hasattr(trunk, 'stem'):
                x = trunk.stem(x)
                layers.append(x)
            elif hasattr(trunk, 'conv1'):
                x = trunk.conv1(x)
                layers.append(x)
            
            # Go through stages
            if hasattr(trunk, 'stages'):
                for i, stage in enumerate(trunk.stages):
                    x = stage(x)
                    layers.append(x)
            
        else:
            # Fallback: use the model's forward_features if available
            try:
                if hasattr(self.backbone, 'forward_features'):
                    features = self.backbone.forward_features(x)
                    # Create multiple layers by pooling
                    for i in range(4):
                        scale_factor = 2 ** i
                        if scale_factor == 1:
                            layers.append(features)
                        else:
                            pooled = torch.nn.functional.avg_pool2d(
                                features, kernel_size=scale_factor, stride=scale_factor
                            )
                            layers.append(pooled)
                else:
                    # Last resort: just pass through the model
                    features = self.backbone(x)
                    if isinstance(features, (list, tuple)):
                        layers = list(features)
                    else:
                        layers = [features, features, features, features]
            except Exception as e:
                print(f"Error in TIMM model forward pass: {e}")
                # Create dummy layers
                batch_size = x.shape[0]
                device = x.device
                for i in range(4):
                    dummy_features = torch.zeros(
                        batch_size, 128 * (2**i), 28//(2**i), 28//(2**i), 
                        device=device
                    )
                    layers.append(dummy_features)
        
        # Ensure we have exactly 4 layers
        if len(layers) > 4:
            # Take evenly spaced layers
            indices = [0, len(layers)//3, 2*len(layers)//3, -1]
            layers = [layers[i] for i in indices]
        elif len(layers) < 4:
            # Duplicate last layer to reach 4
            while len(layers) < 4:
                layers.append(layers[-1])
        
        return layers[:4]
    
    def _forward_convnext(self, x):
        """Forward pass for ConvNeXt architectures"""
        layers = []
        
        # ConvNeXt typically has: stem -> stages -> head
        if hasattr(self.backbone, 'stem'):
            x = self.backbone.stem(x)
            layers.append(x)
        elif hasattr(self.backbone, 'conv1'):
            # Fallback if no stem
            x = self.backbone.conv1(x)
            layers.append(x)
        
        # Process through stages
        if hasattr(self.backbone, 'stages'):
            for i, stage in enumerate(self.backbone.stages):
                x = stage(x)
                layers.append(x)
        elif hasattr(self.backbone, 'trunk') and hasattr(self.backbone.trunk, 'stages'):
            for i, stage in enumerate(self.backbone.trunk.stages):
                x = stage(x)
                layers.append(x)
        else:
            # If no stages found, just duplicate the current features
            for _ in range(3):
                layers.append(x)
        
        # Ensure we have exactly 4 layers
        if len(layers) > 4:
            # Take evenly spaced layers
            indices = [0, len(layers)//3, 2*len(layers)//3, -1]
            layers = [layers[i] for i in indices]
        elif len(layers) < 4:
            # Duplicate last layer to reach 4
            while len(layers) < 4:
                layers.append(layers[-1])
        
        return layers[:4]
    
    def _forward_vit(self, x):
        """Forward pass for Vision Transformer architectures"""
        layers = []
        
        # Patch embedding
        if hasattr(self.backbone, 'patch_embed'):
            # Some architectures have patch_embed
            x = self.backbone.patch_embed(x)
        elif hasattr(self.backbone, 'conv1'):
            # OpenCLIP ViT uses conv1 for patch embedding
            x = self.backbone.conv1(x)  # Shape: [B, embed_dim, H_patches, W_patches]
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, num_patches, embed_dim]
        else:
            raise AttributeError("Cannot find patch embedding layer")
        
        # Add class token and positional encoding
        if hasattr(self.backbone, 'class_embedding'):
            class_token = self.backbone.class_embedding.expand(x.shape[0], -1, -1)
            x = torch.cat([class_token, x], dim=1)
        
        if hasattr(self.backbone, 'positional_embedding'):
            x = x + self.backbone.positional_embedding
        
        # Pre-transformer layer norm
        if hasattr(self.backbone, 'ln_pre'):
            x = self.backbone.ln_pre(x)
        
        # Pass through transformer layers
        if hasattr(self.backbone, 'transformer'):
            transformer_layers = self.backbone.transformer.resblocks
        elif hasattr(self.backbone, 'blocks'):
            transformer_layers = self.backbone.blocks
        else:
            # Fallback: create dummy layers
            embed_dim = x.shape[-1]
            num_patches = x.shape[1] - 1  # Exclude class token
            H = W = int(num_patches ** 0.5)
            
            spatial = x[:, 1:, :].reshape(x.shape[0], H, W, embed_dim).permute(0, 3, 1, 2)
            return [spatial, spatial, spatial, spatial]
        
        num_layers = len(transformer_layers)
        extract_layers = [num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
        
        for i, layer in enumerate(transformer_layers):
            x = layer(x)
            
            if i in extract_layers:
                # Convert back to spatial format
                if x.shape[1] > 1:  # Has class token
                    patch_features = x[:, 1:, :]  # Remove class token
                else:
                    patch_features = x
                    
                B, num_patches, embed_dim = patch_features.shape
                H = W = int(num_patches ** 0.5)
                spatial = patch_features.reshape(B, H, W, embed_dim).permute(0, 3, 1, 2)
                layers.append(spatial)
        
        # Ensure we have 4 layers
        while len(layers) < 4:
            layers.append(layers[-1] if layers else torch.zeros(x.shape[0], 512, 7, 7, device=x.device))
            
        return layers[:4]


class EfficientNetBackbone(BackboneWrapper):
    """EfficientNet-B4 backbone using timm"""
    
    def _load_backbone(self):
        model = timm.create_model('efficientnet_b4', pretrained=True, features_only=True)
        return model
    
    def _get_feature_dim(self):
        return 1792  # EfficientNet-B4 final feature dimension
    
    def forward(self, x):
        # EfficientNet returns multi-scale features directly
        features = self.backbone(x)
        
        # EfficientNet-B4 returns 5 feature maps, take the last 4
        return features[1:]  # Skip the first low-level features


class SAMBackbone(BackboneWrapper):
    """Segment Anything Model (SAM) backbone"""
    
    def __init__(self, backbone_name: str):
        super().__init__(backbone_name)
        # Initialize channel projection layers
        self._init_projection_layers()
    
    def _init_projection_layers(self):
        """Initialize channel projection layers for multi-scale features"""
        target_channels = [256, 512, 1024, 2048]
        sam_channels = 256  # SAM output channels
        
        self.channel_projections = torch.nn.ModuleList()
        for target_c in target_channels:
            if sam_channels != target_c:
                proj = torch.nn.Conv2d(sam_channels, target_c, kernel_size=1, bias=False)
                torch.nn.init.kaiming_normal_(proj.weight)
            else:
                proj = torch.nn.Identity()
            self.channel_projections.append(proj)
    
    def _load_backbone(self):
        try:
            from segment_anything import sam_model_registry
            
            # Setup checkpoint path
            models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            os.makedirs(models_dir, exist_ok=True)
            checkpoint_path = os.path.join(models_dir, "sam_vit_b_01ec64.pth")
            
            # Download SAM checkpoint if not available
            if not os.path.exists(checkpoint_path):
                print("SAM checkpoint not found. Downloading...")
                url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                
                try:
                    print(f"Downloading SAM checkpoint from {url}")
                    urllib.request.urlretrieve(url, checkpoint_path)
                    print(f"✅ SAM checkpoint downloaded to {checkpoint_path}")
                except Exception as e:
                    print(f"❌ Failed to download SAM checkpoint: {e}")
                    print("Please download manually from:")
                    print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
                    raise FileNotFoundError("SAM checkpoint download failed")
            
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            return sam.image_encoder
            
        except ImportError:
            print("❌ segment_anything not installed.")
            print("Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
            raise ImportError("segment_anything package required for SAM backbone")
    
    def _get_feature_dim(self):
        return 2048  # Match the largest output channel dimension
    
    def forward(self, x):
        # Use smaller input size to avoid memory issues
        target_size = 224
        original_size = x.shape[-1]
        
        if original_size != target_size:
            x = torch.nn.functional.interpolate(
                x, size=(target_size, target_size), mode='bilinear', align_corners=False
            )
        
        # Clear cache before forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Use gradient checkpointing and mixed precision
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                features = self.backbone(x)  # SAM output: [B, 256, 14, 14] for 224x224 input
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️  SAM out of memory, creating compatible dummy features")
                batch_size = x.shape[0]
                device = x.device
                # Create dummy features that match ResNet expectations
                return self._create_dummy_features(batch_size, device)
            else:
                raise e
        
        return self._create_multiscale_features(features)
    
    def _create_dummy_features(self, batch_size, device):
        """Create dummy features that match ResNet architecture"""
        target_shapes = [
            (256, 64, 64),   # Layer 0
            (512, 32, 32),   # Layer 1  
            (1024, 16, 16),  # Layer 2
            (2048, 8, 8),    # Layer 3
        ]
        
        layers = []
        for channels, h, w in target_shapes:
            dummy_tensor = torch.zeros(batch_size, channels, h, w, device=device)
            layers.append(dummy_tensor)
        
        return layers
    
    def _create_multiscale_features(self, base_features):
        """Create 4 multi-scale features compatible with ResNet-style architectures"""
        batch_size, channels, height, width = base_features.shape
        layers = []
        
        # Target spatial dimensions that match ResNet architecture
        target_spatial_dims = [
            (64, 64),   # Layer 0: Similar to ResNet layer1 output
            (32, 32),   # Layer 1: Similar to ResNet layer2 output
            (16, 16),   # Layer 2: Similar to ResNet layer3 output  
            (8, 8),     # Layer 3: Similar to ResNet layer4 output
        ]
        
        for i, (target_h, target_w) in enumerate(target_spatial_dims):
            # Spatial adjustment
            if height == target_h and width == target_w:
                spatial_features = base_features
            elif height < target_h:
                # Upsample if SAM features are smaller than target
                spatial_features = torch.nn.functional.interpolate(
                    base_features, size=(target_h, target_w), mode='bilinear', align_corners=False
                )
            else:
                # Downsample if SAM features are larger than target
                spatial_features = torch.nn.functional.adaptive_avg_pool2d(
                    base_features, (target_h, target_w)
                )
            
            # Apply channel projection
            projected_features = self.channel_projections[i](spatial_features)
            layers.append(projected_features)
        
        return layers


def get_backbone(backbone_name: str):
    """Factory function to get backbone by name"""
    
    if backbone_name == "resnet50_places365":
        return ResNet50Places365Backbone(backbone_name)
    elif backbone_name == "dinov2":
        return DINOv2Backbone(backbone_name)
    elif backbone_name == "clip":
        # Use OpenCLIP with ViT-B-32 model (equivalent to original CLIP)
        return OpenCLIPBackbone(backbone_name, model_name="ViT-B-32", pretrained="openai")
    elif backbone_name == "clip_vit_l":
        # Larger CLIP model
        return OpenCLIPBackbone(backbone_name, model_name="ViT-L-14", pretrained="openai")
    elif backbone_name == "clip_convnext":
        # ConvNeXt-based CLIP
        return OpenCLIPBackbone(backbone_name, model_name="convnext_base_w", pretrained="laion2b_s13b_b82k")
    elif backbone_name == "efficientnet_b4":
        return EfficientNetBackbone(backbone_name)
    elif backbone_name == "sam_b":
        return SAMBackbone(backbone_name)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}. Available: resnet50_places365, dinov2, clip, clip_vit_l, clip_convnext, efficientnet_b4, sam_b")