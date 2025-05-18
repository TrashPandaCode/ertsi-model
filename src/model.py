from torch import nn
import torch

class ReverbCNN(nn.Module):
    def __init__(self, num_frequencies):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 1500, 2000)
            conv_out = self.conv(dummy_input)
            self.flat_dim = conv_out.view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, 128),  # Adjust input size
            nn.ReLU(),
            nn.Linear(128, num_frequencies)
        )

    def forward(self, x):
        print("Input shape:", x.shape)
        x = self.conv(x)
        return self.fc(x)