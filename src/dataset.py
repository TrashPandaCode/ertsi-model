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
                T.RandomResizedCrop(224, scale=(0.7, 1.0)),  # More aggressive cropping
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),  # Slightly more rotation
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3
                ),
                T.RandomApply(
                    [T.RandomPerspective(distortion_scale=0.1)], p=0.2
                ),  # Perspective changes
            ]
        else:
            base_transforms += [
                T.CenterCrop(224),
            ]

        base_transforms += [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.transform = transform or T.Compose(base_transforms)

        data_root_list = data_root if isinstance(data_root, list) else [data_root]

        room_dirs = [
            d
            for root in data_root_list
            for d in glob.glob(os.path.join(root, "*"))
            if os.path.isdir(d)
        ]

        # Collect all RT60 values to compute normalization stats
        all_rt60_values = []

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
            all_rt60_values.append(rt60_vector)

            image_paths = glob.glob(os.path.join(room_path, "images", "*.jpg"))
            for image_path in image_paths:
                self.entries.append((image_path, rt60_vector))

        # Compute normalization statistics
        if all_rt60_values:
            all_rt60_tensor = torch.cat(all_rt60_values, dim=0)
            self.rt60_mean = all_rt60_tensor.mean()
            self.rt60_std = all_rt60_tensor.std()
        else:
            self.rt60_mean = 0.0
            self.rt60_std = 1.0

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, rt60 = self.entries[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Normalize RT60 values
        rt60_normalized = (rt60 - self.rt60_mean) / self.rt60_std

        return image, rt60_normalized
