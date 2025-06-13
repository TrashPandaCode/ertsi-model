import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from model_features import RoomSpecificReverbCNN  # Updated import
from dataset_features import RoomAwareReverbDataset  # Updated import
import os

# Settings
DATASET_DIR = "data/train/real"  # or "synth"
MODEL_PATH = "output/reverbcnn.pt"
FREQS = [250, 500, 1000, 2000, 4000, 8000]
USE_TSNE = True  # Set False to use PCA
BATCH_SIZE = 32

# Load dataset with room awareness
dataset = RoomAwareReverbDataset(DATASET_DIR, freqs=FREQS, augment=False, include_room_id=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Get number of rooms for model initialization
num_rooms = len(dataset.room_to_id) if hasattr(dataset, 'room_to_id') else None
print(f"Number of rooms detected: {num_rooms}")

# Load model with correct architecture
model = RoomSpecificReverbCNN(num_frequencies=len(FREQS), num_rooms=num_rooms)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Collect predictions, ground truth, and room information
all_preds, all_targets, all_room_ids = [], [], []
room_names = []

with torch.no_grad():
    for batch in loader:
        if len(batch) == 3:
            imgs, targets, room_ids = batch
            imgs = imgs.to(device)
            room_ids = room_ids.to(device)
            preds = model(imgs, room_ids)
            all_room_ids.append(room_ids.cpu())
        else:
            imgs, targets = batch
            imgs = imgs.to(device)
            preds = model(imgs)
            
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

# Handle room IDs if available
if all_room_ids:
    all_room_ids = torch.cat(all_room_ids).numpy()
    # Create room name mapping for visualization
    id_to_room = {v: k for k, v in dataset.room_to_id.items()}
    room_names = [id_to_room[room_id] for room_id in all_room_ids]
else:
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

# 1. Predictions colored by room (if available)
if all_room_ids is not None:
    unique_rooms = np.unique(all_room_ids)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_rooms)))
    
    for i, room_id in enumerate(unique_rooms):
        mask = all_room_ids == room_id
        room_name = id_to_room[room_id]
        axes[0, 0].scatter(projected_preds[mask, 0], projected_preds[mask, 1], 
                          c=[colors[i]], alpha=0.7, label=room_name, s=30)
    
    axes[0, 0].set_title(f"Predicted RT60s by Room ({method_name})")
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
else:
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
if all_room_ids is not None:
    print(f"Number of unique rooms: {len(unique_rooms)}")
    print(f"Room distribution:")
    for room_id in unique_rooms:
        room_name = id_to_room[room_id]
        count = np.sum(all_room_ids == room_id)
        print(f"  {room_name}: {count} samples")

print(f"\nPrediction Statistics:")
print(f"Prediction range: [{np.min(all_preds):.3f}, {np.max(all_preds):.3f}]")
print(f"Ground truth range: [{np.min(all_targets):.3f}, {np.max(all_targets):.3f}]")
print(f"Overall MSE: {np.mean((all_preds - all_targets)**2):.6f}")
print(f"Overall MAE: {np.mean(np.abs(all_preds - all_targets)):.6f}")