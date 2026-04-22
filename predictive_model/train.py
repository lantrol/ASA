import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim import Adam

# Import simulation components from the main project
from asa import Orientation, Simulation, Transducer
from predictive_model.dataset import get_dataloader
from predictive_model.model import PredModel


class VectorMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred and target shape: (B, 2, 4, 16, 16)
        return F.mse_loss(pred, target)


def train(
    image_dir, label_dir, epochs=10, batch_size=32, lr=1e-3, device="cpu", val_split=0.2
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = PredModel().to(device)

    # DataLoader
    train_loader, val_loader = get_dataloader(
        image_dir, label_dir, batch_size=batch_size, val_split=val_split
    )

    # Loss and Optimizer
    criterion = VectorMSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            inputs_flat = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()
            outputs = model(inputs_flat)
            outputs_reshaped = outputs.view(-1, 2, 4, 16, 16)

            loss = criterion(outputs_reshaped, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(val_loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    inputs_flat = inputs.view(inputs.size(0), -1)
                    outputs = model(inputs_flat)
                    outputs_reshaped = outputs.view(-1, 2, 4, 16, 16)
                    loss = criterion(outputs_reshaped, targets)
                    running_val_loss += loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )
        else:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

    # Final Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Finished Training")
    return model


def run_simulation_on_sample(model, image_path, device="cpu"):
    print(f"Running simulation on specific image: {image_path} using device: {device}")
    model.eval()

    # Simulation parameters (from test_load_bf.py)
    t_mux = 4
    fr = 40e3
    c = 343.0
    sim_dim = 64
    ds = 0.16 / sim_dim

    image = Image.open(image_path).convert("L")  # Ensure single channel
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor / image_tensor.max()
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 64, 64]

    # Predict phases
    with torch.no_grad():
        input_flat = image_tensor.view(1, -1)
        predicted_output = model(input_flat)
        # Reshape to (2, 4, 16, 16) to separate sine and cosine components
        vec_output = predicted_output.view(2, 4, 16, 16).to(device)

        # Reconstruct phases using atan2(sin, cos)
        sin_comp = vec_output[0]
        cos_comp = vec_output[1]
        phases = torch.atan2(sin_comp, cos_comp)

    print(f"Input image: {image_path}")
    print(f"Reconstructed phases shape: {phases.shape}")

    # Simulation setup
    sim = Simulation(fr, c, size=0.16, ds=ds, device=device)

    # We'll just setup one Transducer (Z) as in the test script
    Z = Transducer(
        emitters_num=16,
        array_size=0.16,
        emitter_size=ds,
        apperture=0.01,
        pos=[-0.08, 0, 0],
        orientation=Orientation.Z,
        t_mux=t_mux,
        device=device,
        random_init=False,
    )

    with torch.no_grad():
        Z.phases = phases

    sim.add_transducer(Z)
    sim.create_propagators()
    field = sim.calculate_volume(use_mean=True)

    print("Simulation complete. Field calculated.")
    return field


if __name__ == "__main__":
    IMAGE_DIR = "data/emnist"
    LABEL_DIR = "data/emnist_phases"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Train
    model = train(
        IMAGE_DIR,
        LABEL_DIR,
        epochs=100,
        batch_size=32,
        device=DEVICE,
        lr=0.0001,
        val_split=0.2,
    )

    # Save the model
    model_path = "trained_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 2. Run simulation on a specific image
    field = run_simulation_on_sample(
        model, "data/emnist/8df6fec4-aa76-4a4b-b3ff-f93b1cf30662.png", device=DEVICE
    )

    # Note: napari visualization requires an interactive environment.
    # In a headless CLI, we'll just print success.
    import napari

    print("Launching napari viewer...")
    viewer, layers = napari.imshow(
        torch.abs(field).rot90(1, (2, 1))[10:-10, 10:-10, 10:-10].cpu().detach().numpy()
    )
    napari.run()
