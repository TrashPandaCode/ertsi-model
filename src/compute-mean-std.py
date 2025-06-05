import os
import glob
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class ReverbImageOnlyDataset(Dataset):
    def __init__(self, data_root):
        self.image_paths = []
        # Look recursively for all images in subfolders named "images"
        pattern = os.path.join(data_root, "*", "*", "images", "*.jpg")
        self.image_paths += glob.glob(os.path.join(data_root, "*", "*", "images", "*.jpg"))
        self.image_paths += glob.glob(os.path.join(data_root, "*", "*", "images", "*.png"))

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # or your actual target size
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)

def compute_mean_std(dataset, batch_size=64, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    mean = 0.
    std = 0.
    total_images = 0

    for images in tqdm(loader, desc="Computing mean/std"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)  # Flatten H and W
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std

# Usage
if __name__ == "__main__":
    data_root = r"C:\Users\david\Documents\GitHub\ertsi-model\data\train"
    dataset = ReverbImageOnlyDataset(data_root)
    mean, std = compute_mean_std(dataset)

    print("Dataset Mean: ", mean)
    print("Dataset Std:  ", std)
