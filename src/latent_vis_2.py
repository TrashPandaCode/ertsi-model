import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from model import ReverbCNN  # Updated import
from dataset import ReverbRoomDataset  # Updated import
import os

# Settings
DATASET_DIR = "data/train/real"  # or "synth"
CHECKPOINT_PATH = "checkpoints/real/exV3-reverbcnn-real-epoch=44-val_loss=0.0238.ckpt"  # Best checkpoint
FREQS = [250, 500, 1000, 2000, 4000, 8000]
USE_TSNE = True  # Set False to use PCA
BATCH_SIZE = 32

# Load dataset with room awareness
dataset = ReverbRoomDataset(DATASET_DIR, freqs=FREQS, augment=False)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model from PyTorch Lightning checkpoint
model = ReverbCNN.load_from_checkpoint(CHECKPOINT_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Collect predictions, ground truth, and room information
all_preds, all_targets = [], []
room_names = []

with torch.no_grad():
    for imgs, targets in loader:
        imgs = imgs.to(device)
        preds = model(imgs)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

# Since the dataset doesn't provide room IDs, we'll set this to None
all_room_ids = None

# Create output directory
os.makedirs("latent_vis", exist_ok=True)

# Dimensionality reduction on predictions
data_for_projection = all_preds

if USE_TSNE:
    reducer = TSNE(n_components=2, perplexity=min(30, len(data_for_projection)//4), 
                   init="random", random_state=42)
    projected_preds = reducer.fit_transform(data_for_projection)
    method_name = "t-SNE"
else:
    pca = PCA(n_components=2)
    projected_preds = pca.fit_transform(data_for_projection)
    method_name = "PCA"
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Predictions colored by room (if available) - simplified since no room info
axes[0, 0].scatter(projected_preds[:, 0], projected_preds[:, 1], 
                  alpha=0.6, c="blue", s=30)
axes[0, 0].set_title(f"Predicted RT60s ({method_name})")

axes[0, 0].set_xlabel("Component 1")
axes[0, 0].set_ylabel("Component 2")
axes[0, 0].grid(True, alpha=0.3)

# 2. Predictions colored by average RT60
avg_rt60 = np.mean(all_preds, axis=1)
scatter = axes[0, 1].scatter(projected_preds[:, 0], projected_preds[:, 1], 
                           c=avg_rt60, alpha=0.7, s=30, cmap='viridis')
axes[0, 1].set_title(f"Predicted RT60s by Average Value ({method_name})")
axes[0, 1].set_xlabel("Component 1")
axes[0, 1].set_ylabel("Component 2")
axes[0, 1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[0, 1], label='Average RT60 (s)')

# 3. Ground truth projection
if USE_TSNE:
    reducer_gt = TSNE(n_components=2, perplexity=min(30, len(all_targets)//4), 
                      init="random", random_state=42)
    projected_targets = reducer_gt.fit_transform(all_targets)
else:
    projected_targets = pca.fit_transform(all_targets)

avg_rt60_gt = np.mean(all_targets, axis=1)
scatter_gt = axes[1, 0].scatter(projected_targets[:, 0], projected_targets[:, 1], 
                               c=avg_rt60_gt, alpha=0.7, s=30, cmap='plasma')
axes[1, 0].set_title(f"Ground Truth RT60s ({method_name})")
axes[1, 0].set_xlabel("Component 1")
axes[1, 0].set_ylabel("Component 2")
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter_gt, ax=axes[1, 0], label='Average RT60 (s)')

# 4. Prediction vs Ground Truth comparison
mse_per_sample = np.mean((all_preds - all_targets)**2, axis=1)
scatter_error = axes[1, 1].scatter(projected_preds[:, 0], projected_preds[:, 1], 
                                  c=mse_per_sample, alpha=0.7, s=30, cmap='Reds')
axes[1, 1].set_title(f"Prediction Error ({method_name})")
axes[1, 1].set_xlabel("Component 1")
axes[1, 1].set_ylabel("Component 2")
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter_error, ax=axes[1, 1], label='MSE')

plt.tight_layout()
plt.savefig("latent_vis/comprehensive_latent_visualization.png", dpi=300, bbox_inches='tight')
plt.show()

# Print statistics
print(f"\nDataset Statistics:")
print(f"Number of samples: {len(all_preds)}")
print(f"Number of frequencies: {len(FREQS)}")

print(f"\nPrediction Statistics:")
print(f"Prediction range: [{np.min(all_preds):.3f}, {np.max(all_preds):.3f}]")
print(f"Ground truth range: [{np.min(all_targets):.3f}, {np.max(all_targets):.3f}]")
print(f"Overall MSE: {np.mean((all_preds - all_targets)**2):.6f}")
print(f"Overall MAE: {np.mean(np.abs(all_preds - all_targets)):.6f}")