import os
import sys
from pathlib import Path

import napari
import numpy as np
import torch
from PIL import Image

from asa import Orientation, Simulation_Batch, Transducer_Batch

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parents[1]))
#
from predictive_model_pinn.model_sin_cos import SinCosModel as PredModel

# from predictive_model_pinn.model_resid import ResidModel as PredModel

# from predictive_model_pinn.model_multi import MultiModel as PredModel

# from predictive_model_pinn.model_deep import DeeperModel as PredModel


def run_inference_and_simulate(model_path, image_paths, device="cpu"):
    print(f"Loading model from {model_path}...")
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = PredModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    print(f"Loading image from {image_paths}...")

    all_images = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("L")
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor / image_tensor.max()
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
    MODEL_PATH = "trained_model_pinn_rand.pth"
    IMAGE_PATH1 = "samples/smiley_64.png"
    IMAGE_PATH2 = "data/emnist/8df6fec4-aa76-4a4b-b3ff-f93b1cf30662.png"
    IMAGE_PATH3 = "samples/birb.png"
    IMAGE_PATH4 = "samples/loss.png"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    field = run_inference_and_simulate(
        MODEL_PATH, [IMAGE_PATH1, IMAGE_PATH2, IMAGE_PATH3, IMAGE_PATH4], device=DEVICE
    )

    print("Launching napari viewer...")
    # Visualize the absolute value of the field, cropped for better viewing
    vis_data = torch.abs(field).cpu().detach().numpy()
    napari.imshow(vis_data)
    napari.run()
