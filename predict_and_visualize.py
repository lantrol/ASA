import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
import torch
from matplotlib.typing import JoinStyleType
from PIL import Image
from torch import nn

from asa import Orientation, Simulation_Batch, Transducer_Batch
from predictive_model_pinn.model_factory import get_model_by_name

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parents[1]))
#


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


def run_inference_and_simulate(model_type, model_path, image_paths, device="cpu"):
    print(f"Loading model from {model_path}...")
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = get_model_by_name(model_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    print(f"Loading image from {image_paths}...")

    all_images = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("L")
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = (image_tensor - image_tensor.min()) / (
            image_tensor.max() - image_tensor.min()
        )
        image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 64, 64]
        all_images.append(image_tensor)

    input_images = torch.stack(all_images, dim=0)

    # Simulation parameters
    t_mux = 4
    fr = 40e3
    c = 343.0
    sim_dim = 64
    ds = 0.16 / sim_dim

    # Predict phases
    with torch.no_grad():
        input_flat = input_images.view(input_images.size(0), -1)
        phases = model(input_flat).view(-1, 4, 16, 16)

        # predicted_output shape is [1, 2, 4, 16, 16] -> (B, 2, 4, 16, 16)
        # Split sin and cos
        # - sin_comp = predicted_output[:, :, :, :, 0]
        # - cos_comp = predicted_output[:, :, :, :, 1]
        # Reconstruct phases using atan2
        # - phases = torch.atan2(sin_comp, cos_comp)  # [1, 4, 16, 16]
        # phases = phases.squeeze(0)  # [4, 16, 16]

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
    sim.create_propagator_slices(0.08)
    field = sim.calculate_slices(use_mean=True).reshape(len(image_paths), 64, 64)

    print(field.shape)

    print("Simulation complete. Field calculated.")
    return field


if __name__ == "__main__":
    IMAGE_PATH1 = "samples/smiley_64.png"
    IMAGE_PATH2 = "samples/domino.png"
    IMAGE_PATH3 = "samples/birb.png"
    IMAGE_PATH4 = "samples/loss.png"
    IMAGE_PATH5 = "data/val_images/fe5a0db9-0ee7-4323-8796-6120deb2a458.png"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model_paths = [
        "checkpoints/CosineLoss_20260504_125523_best.pth",
        "checkpoints/noise_CosineLoss_20260504_142512_best.pth",
        "best_model_pinn_rand.pth",
        "model_random_noise.pth",
        # "checkpoints/ScaleInvariantMSE_20260504_132232_best.pth",
        # "checkpoints/NormalizedMSE_20260504_130217_best.pth",
    ]

    # model_paths = [
    #     "checkpoints/noise_CosineLoss_20260504_110656_best.pth",
    #     "checkpoints/noise_ScaleInvariantMSE_20260504_111556_best.pth",
    #     "checkpoints/noise_NormalizedMSE_20260504_111406_best.pth",
    # ]

    all_images = [IMAGE_PATH1, IMAGE_PATH2, IMAGE_PATH3, IMAGE_PATH4, IMAGE_PATH5]

    criterion = ScaleInvariantMSELoss()

    num_fields = len(all_images)
    for i, path in enumerate(all_images):
        image = Image.open(path).convert("L")
        plt.subplot(len(model_paths) + 1, num_fields, i + 1)
        plt.imshow(image)

    for i, model_path in enumerate(model_paths):
        field = run_inference_and_simulate(
            "SinCosModel",
            model_path,
            all_images,
            device=DEVICE,
        )

        vis_data = torch.abs(field).cpu().detach().numpy()

        for j in range(num_fields):
            image = Image.open(all_images[j]).convert("L")
            image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            image_tensor = (image_tensor - image_tensor.min()) / (
                image_tensor.max() - image_tensor.min()
            )
            image_tensor = image_tensor.unsqueeze(0).to("cuda")

            print(
                criterion(torch.tensor(vis_data[j, ...], device="cuda"), image_tensor)
            )

            plt.subplot(len(model_paths) + 1, num_fields, (i + 1) * num_fields + j + 1)
            plt.imshow(vis_data[j, ...])

    plt.show(block=True)
