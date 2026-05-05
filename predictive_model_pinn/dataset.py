import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class PredictiveDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = sorted(
            glob.glob(os.path.join(image_dir, "*.[jp][pn][g]"))
            + glob.glob(os.path.join(image_dir, "*.bmp"))
            + glob.glob(os.path.join(image_dir, "*.tiff"))
        )

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]

        # Load image
        image = Image.open(img_path).convert("L")  # Ensure single channel
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image / image.max()
            image = image.unsqueeze(0)  # Add channel dimension

        return image


class NoiseDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]

        # Load image
        image = Image.open(img_path).convert("L")  # Ensure single channel
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image / image.max()
            image = image.unsqueeze(0)  # Add channel dimension

        return image


def get_dataloader(image_dir, val_dir, batch_size=32, shuffle=True, val_split=0.2):
    train_dataset = PredictiveDataset(image_dir)
    val_dataset = PredictiveDataset(val_dir)

    if val_split > 0:
        # val_size = int(len(dataset) * val_split)
        # train_size = len(dataset) - val_size
        # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        return train_loader, val_loader
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
        )
        return train_loader, None
