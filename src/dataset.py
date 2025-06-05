import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
from PIL import Image
import torch


class ReverbRoomDataset(Dataset):
    def __init__(self, data_root, transform=None, freqs=None, augment=True):
        self.entries = []
        self.freqs = freqs
        self.augment = augment

        base_transforms = [
            T.Resize((256, 256)),
        ]

        if augment:
            base_transforms += [
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
                T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
            ]
        else:
            base_transforms += [
                T.CenterCrop(224),
            ]

        base_transforms += [
            T.ToTensor(),
            T.Normalize(mean=[0.3942, 0.3814, 0.3651], std=[0.1533, 0.1538, 0.1556]),
        ]

        self.transform = transform or T.Compose(base_transforms)

        room_dirs = [
            d for d in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(d)
        ]

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
