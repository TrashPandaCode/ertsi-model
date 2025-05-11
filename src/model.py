from torch import nn

class ReverbCNN(nn.Module):
    def __init__(self, num_frequencies):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 128),  # Adjust input size
            nn.ReLU(),
            nn.Linear(128, num_frequencies)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)