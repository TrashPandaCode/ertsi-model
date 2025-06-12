import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from model_flex import ReverbCNN
from simple_model import SimpleReverbCNN
from dataset import ReverbRoomDataset

# Settings
DATASET_DIR = "data/train/synth/hybrid/"  # or "synth"
MODEL_PATH = "output/comparison-reverbcnn_resnet50_places365.pt"
FREQS = [250, 500, 1000, 2000, 4000, 8000]
USE_TSNE = True  # Set False to use PCA
BATCH_SIZE = 32

# Load dataset
dataset = ReverbRoomDataset(DATASET_DIR, freqs=FREQS, augment=False)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
# model = ReverbCNN(num_frequencies=len(FREQS), backbone="resnet50_places365")
# model.load_state_dict(torch.load(MODEL_PATH))
model = SimpleReverbCNN.load_from_checkpoint("output/simple_reverbcnn.ckpt")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Collect predictions and ground truth
all_preds, all_targets = [], []

with torch.no_grad():
    for imgs, targets in loader:
        imgs = imgs.to(device)
        preds = model(imgs)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

# Dimensionality reduction
data_for_projection = all_preds  # or all_targets

if USE_TSNE:
    reducer = TSNE(n_components=2, perplexity=30, init="random", random_state=42)
    projected = reducer.fit_transform(data_for_projection)
else:
    pca = PCA(n_components=2)
    projected = pca.fit_transform(data_for_projection)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(
    projected[:, 0], projected[:, 1], alpha=0.6, c="blue", label="Predicted RT60s"
)
plt.title("t-SNE" if USE_TSNE else "PCA")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
