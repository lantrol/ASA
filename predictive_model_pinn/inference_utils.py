import torch
import numpy as np
from PIL import Image

def run_simulation_on_sample(model, image_path, device="cpu"):
    print(f"Running simulation on specific image: {image_path} using device: {device}")
    model.eval()

    t_mux = 4
    fr = 40e3
    c = 343.0
    sim_dim = 64
    ds = 0.16 / sim_dim

    from asa import Orientation, Simulation_Batch, Transducer_Batch

    image = Image.open(image_path).convert("L")
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor / image_tensor.max()
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        input_flat = image_tensor.view(1, -1)
        predicted_output = model(input_flat)
        vec_output = predicted_output.view(1, 4, 16, 16, 2).to(device)

        sin_comp = vec_output[:, :, :, :, 0]
        cos_comp = vec_output[:, :, :, :, 1]
        phases = torch.atan2(sin_comp, cos_comp).squeeze(0)

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

    with torch.no_grad():
        Z.phases = phases

    sim.add_transducer(Z)
    sim.create_propagators()
    field = sim.calculate_volume(use_mean=True)

    print("Simulation complete. Field calculated.")
    return field
