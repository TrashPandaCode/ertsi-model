import os
import torch
from torch import nn
from torchvision import models

base_dir = os.path.dirname(os.path.abspath(__file__))


def resnet50_places365():
    model_path = os.path.join(base_dir, "..", "models", "resnet50_places365.pth.tar")
    model = models.resnet50(weights=None)

    state_dict = strip(torch.load(model_path, map_location="cpu"))

    model.load_state_dict(state_dict, strict=False)
    return model


def resnet18_places365():
    model_path = os.path.join(base_dir, "..", "models", "resnet18_places365.pth.tar")
    model = models.resnet18(weights=None)

    state_dict = strip(torch.load(model_path, map_location="cpu"))

    model.load_state_dict(state_dict, strict=False)
    return model


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
