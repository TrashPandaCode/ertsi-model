import torch
from model import ReverbCNN
from torchvision import transforms
from PIL import Image

# 1. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load model
model = ReverbCNN(num_frequencies=6).to(device)

model.eval()

# 3. Create dummy image (or load one real image)
dummy_image = torch.randn(1, 3, 224, 224).to(device)  # batch of 1, 3x224x224

# 4. Run model
with torch.no_grad():
    output = model(dummy_image)

# 5. Print shape
print("Output shape:", output.shape)  # should be [1, 6] if num_outputs = 6