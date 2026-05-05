import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim import Adam
from torch.optim.adamw import AdamW
from torch.utils.tensorboard import SummaryWriter

# Import simulation components from the main project
from asa import Orientation, Simulation_Batch, Transducer_Batch
from predictive_model_pinn.dataset import get_dataloader

torch.manual_seed(1)
np.random.seed(2)
random.seed(3)


def get_inverse_sqrt_schedule(optimizer, warmup_steps, last_epoch=-1):
    def lr_lambda(step):
        step = max(step, 1)
        if step < warmup_steps:
            return step / warmup_steps
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
        alpha = torch.dot(pred_flat, target_flat) / (
            torch.dot(pred_flat, pred_flat) + self.eps
        )
        pred_scaled = alpha * pred_flat
        loss = ((pred_scaled - target_flat) ** 2).mean()
        return loss


class Normalized_MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min())
        return ((pred_norm - target) ** 2).mean()


def train(
    model,
    image_dir,
    label_dir,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    criterion=None,
    device="cpu",
    val_split=0.2,
    writer=None,
    experiment_name="experiment",
    save_path="best_model.pth",
    flatten=True,
    noise=True,
):
    if criterion is None:
        criterion = CosineLoss()

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

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

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_inverse_sqrt_schedule(optimizer, 20 * len(train_loader))

    train_losses = []
    val_losses = []

    patience = 20
    counter = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for i, inputs in enumerate(train_loader):
            # Simulation-based training logic (keeping existing logic)
            if noise:
                rand_phases = (
                    torch.rand(
                        (batch_size, 4, 16, 16), dtype=torch.float32, device=device
                    )
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
                    torch.rand(
                        inputs.shape[0], dtype=torch.float32, device=device
                    ).expand(inputs.shape)
                    * 0.6
                    + 0.2
                )
                inputs[inputs > threshold] = 1
                inputs[inputs < threshold] = 0
            else:
                inputs = inputs.to(device)

            inputs_flat = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()
            phases = model(inputs_flat if flatten else inputs)

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

        if writer:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)

        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for i, inputs in enumerate(val_loader):
                    inputs = inputs.to(device)

                    inputs_flat = inputs.view(inputs.size(0), -1)

                    phases = model(inputs_flat if flatten else inputs)

                    with torch.no_grad():
                        sim.transducers[0].phases = phases

                    fields = sim.calculate_slices(use_mean=True).reshape(
                        inputs.size(0), 1, 64, 64
                    )

                    loss = criterion(fields, inputs)
                    running_val_loss += loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            if writer:
                writer.add_scalar("Loss/val", avg_val_loss, epoch)

            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        else:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

    print("Finished Training")
    return train_losses, val_losses
