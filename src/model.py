import torch
import load_model
from torch import nn
from torchvision import models


class ReverbCNN(nn.Module):
    def __init__(self, num_frequencies=6):
        super().__init__()

        base = load_model.resnet50_places365()
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),   #(512, 256) for ResNet18
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_frequencies)
        )

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.backbone(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x