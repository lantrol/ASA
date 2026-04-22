import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class PredictiveDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
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

        # Load label
        # image name: path/to/img.png -> label path: label_dir/img.png.txt
        base_name = os.path.basename(img_path)
        label_path = os.path.join(self.label_dir, base_name + ".txt")

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        # Load 4 rows of 16*16 numbers
        # image name: path/to/img.png -> label path: label_dir/img.png.txt
        base_name = os.path.basename(img_path)
        label_path = os.path.join(self.label_dir, base_name + ".txt")

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        # Load 4 rows of 16*16 numbers
        label = np.loadtxt(label_path).reshape(4, 16, 16)
        label = torch.from_numpy(label).float()

        # Bound all phases in the range [0, 2*pi)
        label = torch.remainder(label, 2 * np.pi)

        # Convert phase labels into (sin, cos) pairs
        sin_label = torch.sin(label)
        cos_label = torch.cos(label)

        # Stack sin and cos: shape (2, 4, 16, 16)
        label = torch.stack([sin_label, cos_label], dim=0)

        return image, label


def get_dataloader(image_dir, label_dir, batch_size=32, shuffle=True, val_split=0.2):
    dataset = PredictiveDataset(image_dir, label_dir)

    if val_split > 0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader, None
