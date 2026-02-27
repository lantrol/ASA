import math
import os
import time

import glm
import matplotlib.pyplot as plt
import napari
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm

from asa import (
    Orientation,
    Simulation_ASA_Spectrum,
    SimulationASA,
    Transducer,
    Transducer_Spectrum,
)
from volume_utils import create_donut, cube_frame, helix, hollow_sphere, vowl

torch.manual_seed(68)

t_mux = 16
fr = 40e3
c = 343.0
wavelen = c / fr
sim_dim = 64
ds = 0.16 / sim_dim

print(sim_dim)

device = "cuda"

sim = Simulation_ASA_Spectrum(fr, c, size=0.16, ds=ds, device=device)


sim.add_transducer(
    Transducer_Spectrum(
        16, 0.16, 0.008, [-0.1, 0, 0], Orientation.X, t_mux=t_mux, device=device
    )
)
sim.add_transducer(
    Transducer_Spectrum(
        16, 0.16, 0.008, [-0.1, 0, 0], Orientation.Y, t_mux=t_mux, device=device
    )
)
sim.add_transducer(
    Transducer_Spectrum(
        16, 0.16, 0.008, [-0.1, 0, 0], Orientation.Z, t_mux=t_mux, device=device
    )
)

sim.create_propagators()

image = Image.open("./samples/domino.png").convert("L")
sample = torch.tensor(np.asarray(image)[:, :], device=device)
sample = sample / sample.max()

image2 = Image.open("./samples/smiley.png").convert("L")
sample2 = torch.tensor(np.asarray(image2)[::2, ::2], device=device)
sample2 = sample2 / sample2.max()


image3 = Image.open("./samples/loss.png").convert("L")
sample3 = torch.tensor(np.asarray(image3)[:, :], device=device)
sample3 = sample3 / sample3.max()


sample = torch.roll(create_donut(sim_dim, 3, 16), 6, 0)
sample += torch.roll(create_donut(64, 3, 14), -7, 0).rot90(1, (1, 2))

# sample = cube_frame(64, r=0.03, R=0.045, rot=[0, 0, 0])
# sample = helix()
# sample += helix(rot=[0, 0, math.pi])
sample = sample.to(device)

viewer, layers = napari.imshow(torch.abs(sample).cpu().detach().numpy())
napari.run()

params = []
for tr in sim.transducers:
    params.append(tr.phases)
    params.append(tr.amps)


start = time.time()

iters = 400

optimizer = torch.optim.Adam(params, lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 600, gamma=0.9)
losses = []
for k in (pbar := tqdm(range(iters))):
    field = sim.calculate_volume_spectrum()
    field = field.abs()

    # Volume
    # loss = ((field - sample) ** 2).sum() / field.numel()

    loss = 1 - torch.dot(field.flatten(), sample.flatten()).sum() ** 2 / (
        (field**2).sum() * (sample**2).sum() + 1e-8
    )

    loss.backward()

    # Slice a bit bruteforce-y
    # loss = ((field[0, :, :] / field[0, :, :].max() - sample) ** 2).sum()
    # loss = ((field[0, :, :] / field[16, :, :].max() - sample2) ** 2).sum()
    # loss += ((field[63, :, :] / field[32, :, :].max() - sample3) ** 2).sum()
    # loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    # scheduler.step()

    losses.append(loss.item())
    pbar.set_description(str(losses[-1]))


duration = time.time() - start
print(f"Total optimization time for {iters} iterations: ", duration)
print(f"Final loss: {losses[-1]:.2f}")

plt.plot(range(iters), losses)
plt.ylim((0, 1))
plt.show(block=True)

field = sim.calculate_volume_spectrum()
# plt.imshow(field[0, :, :].cpu().detach().numpy())
# plt.show(block=True)

# plt.imshow(field[32, :, :].cpu().detach().numpy())
# plt.show(block=True)

# plt.imshow(field[63, :, :].cpu().detach().numpy())
# plt.show(block=True)

print(sim.transducers[0].amps[:, 0, 0])

viewer, layers = napari.imshow(
    torch.abs(field).rot90(-0, (0, 1)).cpu().detach().numpy()
)
napari.run()
