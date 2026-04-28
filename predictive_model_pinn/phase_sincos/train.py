import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import os
import random
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim import Adam

# Import simulation components from the main project
from asa import Orientation, Simulation_Batch, Transducer_Batch
from predictive_model_pinn.dataset import get_dataloader

torch.manual_seed(1)
np.random.seed(2)
random.seed(3)

LOG_TRAIN = True


class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = 1 - torch.dot(pred.flatten(), target.flatten()).sum() ** 2 / (
            (pred**2).sum() * (target**2).sum() + 1e-8
        )
        return loss


def train(
    model,
    image_dir,
    label_dir,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    criterion=CosineLoss(),
    device="cpu",
    val_split=0.2,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = model.to(device)

    # DataLoader
    train_loader, val_loader = get_dataloader(
        image_dir, label_dir, batch_size=batch_size, val_split=val_split
    )

    # Simulation setup
    t_mux = 4
    fr = 40e3
    c = 343.0
    sim_dim = 64
    ds = 0.16 / sim_dim
    sim = Simulation_Batch(fr, c, size=0.16, ds=ds, device=device)

    # We'll just setup one Transducer (Z) as in the test script
    Z = Transducer_Batch(
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

    sim.add_transducer(Z)
    sim.create_propagator_slices(0.08)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)  # weight_decay=1e-4
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )

    train_losses = []
    val_losses = []

    # Early Stopping parameters
    patience = 20
    counter = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            # p_rand = torch.rand(1)

            # if p_rand > 1 - (epoch / epochs) * 0.3:
            #     inputs = inputs.to(device)
            # targets = targets.to(device)

            # else:
            #     img_dim = random.choice([16, 24, 32])
            #     inputs = torch.rand(
            #         (batch_size, 1, img_dim, img_dim),
            #         dtype=torch.float32,
            #         device=device,
            #     )

            #     if img_dim != 64:
            #         inputs = F.interpolate(
            #             inputs,
            #             size=(64, 64),
            #             mode="bilinear",  # , align_corners=False
            #         )

            #     inputs[inputs > 0.5] = 1
            #     inputs[inputs < 0.5] = 0

            # for j in range(4):
            #     if random.random() > 0.5:
            #         ini, fini = batch_size // 4 * j, batch_size // 4 * (j + 1)
            #         inputs[ini:fini, :, :, :][inputs[ini:fini] > 0.5] = 1
            #         inputs[ini:fini, :, :, :][inputs[ini:fini] < 0.5] = 0

            # ---
            # rand_phases = (
            #     torch.rand((batch_size, 4, 16, 16), dtype=torch.float32, device=device)
            #     * 2
            #     * torch.pi
            # )
            # with torch.no_grad():
            #     sim.transducers[0].phases = rand_phases
            #     inputs = sim.calculate_slices(use_mean=True).reshape(
            #         batch_size, 1, 64, 64
            #     )

            inputs = inputs.to(device)
            inputs_flat = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()
            phases = model(inputs_flat)

            with torch.no_grad():
                sim.transducers[0].phases = phases

            fields = sim.calculate_slices(use_mean=True).reshape(
                inputs.size(0), 1, 64, 64
            )

            loss = criterion(fields, inputs)
            loss.backward()
            optimizer.step()
            scheduler.step()

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

                    phases = model(inputs_flat)

                    with torch.no_grad():
                        sim.transducers[0].phases = phases

                    fields = sim.calculate_slices(use_mean=True).reshape(
                        inputs.size(0), 1, 64, 64
                    )

                    loss = criterion(fields, inputs)
                    running_val_loss += loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )
            # Early Stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                # Optionally save the model here
                torch.save(model.state_dict(), "best_model_pinn_rand.pth")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
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

    print(f"Finished Training: Best Val loss of {np.array(val_losses).min()}")
    return train_losses, val_losses


# Moved import here for easier modifying
from predictive_model_pinn.models.model_phase import PhaseModel
from predictive_model_pinn.models.model_sin_cos import SinCosModel

if __name__ == "__main__":
    IMAGE_DIR = "data/emnist"
    LABEL_DIR = "data/emnist_phases"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    lr = 0.0001
    criterion = CosineLoss()

    models = [PhaseModel, SinCosModel]

    for model_type in models:
        model = model_type()

        # 1. Train
        train_loss, val_loss = train(
            model,
            IMAGE_DIR,
            LABEL_DIR,
            epochs=25,
            batch_size=64,
            device=DEVICE,
            lr=lr,
            criterion=criterion,
            val_split=0.2,
        )

        # Save the model
        model_path = "trained_model_pinn_rand.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        if LOG_TRAIN:
            utc_time = datetime.now(timezone.utc)
            info = {
                "base_lr": lr,
                "model": model.__class__.__name__,
                "loss_function": criterion.__class__.__name__,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "date": str(utc_time),
            }

            with open(
                f"{Path(__file__).resolve().parents[0]}/{model.__class__.__name__}.json",
                "w",
            ) as json_file:
                json.dump(info, json_file, indent=4)
