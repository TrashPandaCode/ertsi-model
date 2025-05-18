import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
from PIL import Image
import torch

class ReverbRoomDataset(Dataset):
    def __init__(self, data_root, transform=None, freqs=None):
        self.entries = []
        self.freqs = freqs
        self.transform = transform or T.Compose([
            T.Resize((1500,2000)),
            T.ToTensor()
        ])

        room_dirs = [d for d in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(d)]

        for room_path in room_dirs:
            rt60_path = os.path.join(room_path, "rt60.csv")
            
            try:
                df = pd.read_csv(rt60_path)
            except FileNotFoundError:
                print(f"RT60 file not found in {room_path}. Skipping this directory.")
                continue

            if self.freqs:
                df = df[df["Frequency (Hz)"].isin(self.freqs)]
            
            rt60_vector = torch.tensor(df["RT60 (s)"].values, dtype=torch.float32)

            image_paths = glob.glob(os.path.join(room_path, "images", "*.jpg"))
            for image_path in image_paths:
                self.entries.append((image_path, rt60_vector))
    
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, rt60 = self.entries[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, rt60