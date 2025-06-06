import os
import torch
from torch import nn
from torchvision import models
import timm

base_dir = os.path.dirname(os.path.abspath(__file__))


def resnet50_places365():
    model_path = os.path.join(base_dir, "..", "models", "resnet50_places365.pth.tar")
    model = models.resnet50(weights=None)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = strip(torch.load(model_path, map_location=device))
    model.load_state_dict(state_dict, strict=False)

    # Move model to device
    model = model.to(device)
    return model


def resnet18_places365():
    model_path = os.path.join(base_dir, "..", "models", "resnet18_places365.pth.tar")
    model = models.resnet18(weights=None)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = strip(torch.load(model_path, map_location=device))
    model.load_state_dict(state_dict, strict=False)

    # Move model to device
    model = model.to(device)
    return model


def efficientnet_b4():
    """EfficientNet-B4 - excellent balance of accuracy and efficiency"""
    model = timm.create_model("tf_efficientnet_b4", pretrained=True)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


def convnext_base():
    """ConvNeXt-Base - modern CNN with hierarchical features"""
    model = timm.create_model("convnext_base", pretrained=True)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


def swin_base():
    """Swin Transformer - hierarchical vision transformer"""
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


def densenet169():
    """DenseNet169 - dense connections for feature reuse"""
    model = models.densenet169(weights="IMAGENET1K_V1")

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


def get_model_info(model_name):
    """Get model and its feature dimensions"""
    if model_name == "resnet50_places365":
        model = resnet50_places365()
        backbone = nn.Sequential(*list(model.children())[:-2])
        feature_dim = 2048
        needs_pool = True

    elif model_name == "efficientnet_b4":
        model = efficientnet_b4()
        # EfficientNet uses forward_features method
        backbone = lambda x: model.forward_features(x)
        feature_dim = 1792  # EfficientNet-B4 feature dimension
        needs_pool = True

    elif model_name == "convnext_base":
        model = convnext_base()
        backbone = lambda x: model.forward_features(x)
        feature_dim = 1024  # ConvNeXt-Base feature dimension
        needs_pool = True

    elif model_name == "swin_base":
        model = swin_base()
        backbone = lambda x: model.forward_features(x)
        feature_dim = 1024  # Swin-Base feature dimension
        needs_pool = True

    elif model_name == "densenet169":
        model = densenet169()
        backbone = model.features
        feature_dim = 1664  # DenseNet169 feature dimension
        needs_pool = True

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Ensure backbone is on the same device as the model
    device = next(model.parameters()).device
    if hasattr(backbone, "to"):
        backbone = backbone.to(device)

    return backbone, feature_dim, needs_pool


def strip(state_dict):
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    for k in list(new_state_dict.keys()):
        if k.startswith("fc."):
            del new_state_dict[k]

    return new_state_dict
