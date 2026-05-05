import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import random
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# Import simulation components from the main project
from asa import Orientation, Simulation_Batch, Transducer_Batch
from predictive_model_pinn.dataset import get_dataloader


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


def train_vae(
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
    experiment_name="vae_experiment",
    save_path="best_vae_model.pth",
    beta=0.01,
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

    optimizer = Adam(model.parameters(), lr=lr)
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

            optimizer.zero_grad()

            # VAE forward call: (phases, mu, logvar)
            phases, mu, logvar = model(inputs, training=True)

            with torch.no_grad():
                sim.transducers[0].phases = phases

            fields = sim.calculate_slices(use_mean=True).reshape(
                inputs.size(0), 1, 64, 64
            )

            loss_sim = criterion(fields, inputs)
            loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = loss_sim + (beta * loss_kl)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_train_loss += loss.item()

            if writer:
                writer.add_scalar(
                    "Loss/kl", loss_kl.item(), epoch * len(train_loader) + i
                )

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
