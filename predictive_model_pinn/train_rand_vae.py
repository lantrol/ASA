import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

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

LOG_TRAIN = False


def get_inverse_sqrt_schedule(optimizer, warmup_steps, last_epoch=-1):
    def lr_lambda(step):
        step = max(step, 1)

        # Linear warmup
        if step < warmup_steps:
            return step / warmup_steps

        # Inverse sqrt decay
        return (warmup_steps**0.5) / (step**0.5)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=last_epoch
    )


class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = 1 - torch.dot(pred.flatten(), target.flatten()).sum() ** 2 / (
            (pred**2).sum() * (target**2).sum() + 1e-8
        )
        return loss


class ScaleInvariantMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        # optimal scaling factor
        alpha = torch.dot(pred_flat, target_flat) / (
            torch.dot(pred_flat, pred_flat) + self.eps
        )

        pred_scaled = alpha * pred_flat

        loss = ((pred_scaled - target_flat) ** 2).mean()
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
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = get_inverse_sqrt_schedule(optimizer, 20 * len(train_loader))

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
            # inputs = inputs.to(device)
            # targets = targets.to(device)

            rand_phases = (
                torch.rand((batch_size, 4, 16, 16), dtype=torch.float32, device=device)
                * 2
                * torch.pi
            )
            with torch.no_grad():
                sim.transducers[0].phases = rand_phases
                inputs = sim.calculate_slices(use_mean=True).reshape(
                    batch_size, 1, 64, 64
                )
                inputs = inputs / inputs.max()

            threshold = (
                torch.rand(inputs.shape[0], dtype=torch.float32, device=device).expand(
                    inputs.shape
                )
                * 0.6
                + 0.2
            )
            inputs[inputs > threshold] = 1
            inputs[inputs < threshold] = 0

            # VAE hyperparams
            beta = 0.01  # KL weight
            recon_weight = 0.0  # Reconstruction weight

            optimizer.zero_grad()

            # New forward call for training
            phases, mu, logvar = model(inputs, training=True)

            with torch.no_grad():
                sim.transducers[0].phases = phases

            fields = sim.calculate_slices(use_mean=True).reshape(
                inputs.size(0), 1, 64, 64
            )

            # 1. Simulation Loss (End-to-End)
            loss_sim = criterion(fields, inputs)

            # 2. VAE Reconstruction Loss
            # loss_recon = F.mse_loss(reconstruction, inputs)

            # 3. VAE KL Divergence Loss
            loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Total Loss
            loss = loss_sim + (beta * loss_kl)

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

                    # Use the non-training forward call
                    phases = model(inputs, training=False)

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
    plt.savefig("transformernr.png")

    print("Finished Training")
    return train_losses, val_losses


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
        # Use the non-training forward call
        phases = model(image_tensor, training=False)

    print(f"Input image: {image_path}")
    print(f"Reconstructed phases shape: {phases.shape}")

    # Simulation setup
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

    with torch.no_grad():
        Z.phases = phases

    sim.add_transducer(Z)
    sim.create_propagators()
    field = sim.calculate_volume(use_mean=True)

    print("Simulation complete. Field calculated.")
    return field


from predictive_model_pinn.models.model_vae_v3 import PatchVAETransformer as PredModel

if __name__ == "__main__":
    IMAGE_DIR = "data/emnist"
    LABEL_DIR = "data/emnist_phases"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    lr = 0.001
    criterion = CosineLoss()
    model = PredModel()

    # 1. Train
    train_loss, val_loss = train(
        model,
        IMAGE_DIR,
        LABEL_DIR,
        epochs=250,
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
        info = {
            "base_lr": lr,
            "loss_function": criterion.__name__,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        utc_time = datetime.now(timezone.utc)

        with open(f"train_runs/noise_trained_{utc_time}.json", "w") as json_file:
            json.dump(info, json_file, indent=4)
