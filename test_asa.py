import math
import os
import time

import matplotlib.pyplot as plt
import napari
import numpy as np
import torch
from PIL import Image

from asa import Orientation, SimulationASA, Transducer

torch.manual_seed(68)

sim_dim = 64
t_mux = 16
fr = 40e3
c = 343.0

device = "cuda"

sim = SimulationASA(fr, c, size=0.16, ds=0.16 / sim_dim, device=device)


sim.add_transducer(
    Transducer(
        16, 0.16, 0.008, [-0.12, 0, 0], Orientation.X, t_mux=t_mux, device=device
    )
)
sim.add_transducer(
    Transducer(
        16, 0.16, 0.008, [-0.12, 0, 0], Orientation.Y, t_mux=t_mux, device=device
    )
)
sim.add_transducer(
    Transducer(
        16, 0.16, 0.008, [-0.12, 0, 0], Orientation.Z, t_mux=t_mux, device=device
    )
)

sim.create_propagators()


# sigma = sim_dim / 12  # controls spread (smaller = sharper sphere)
# # Create coordinate grid
# coords = torch.linspace(-1, 1, sim_dim)
# x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")
# # Squared distance from center
# r2 = x**2 + y**2 + z**2
# # Gaussian
# sample = torch.exp(-r2 / (2 * (sigma / sim_dim) ** 2)).to(device) * 2

image = Image.open("./samples/domino.png").convert("L")
sample = torch.tensor(np.asarray(image)[:, :], device=device)
sample = sample / sample.max()
print(sample.shape)

image2 = Image.open("./samples/smiley.png").convert("L")
sample2 = torch.tensor(np.asarray(image2)[::2, ::2], device=device)
sample2 = sample2 / sample2.max()

image3 = Image.open("./samples/loss.png").convert("L")
sample3 = torch.tensor(np.asarray(image3)[:, :], device=device)
sample3 = sample3 / sample3.max()
# plt.imshow(sample.cpu().detach().numpy())
# plt.show(block=True)

params = []
for tr in sim.transducers:
    params.append(tr.phases)
    params.append(tr.amps)


start = time.time()
iters = 1000
optimizer = torch.optim.AdamW(params, lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 300, gamma=0.9)
for k in range(iters):
    field = sim.calculate_volume()
    field = field.abs()  # / torch.abs(field).max()

    # Volume
    # loss = ((field - sample) ** 2).sum()
    # loss.backward()

    # Slice a bit bruteforce-y
    loss = ((field[0, :, :] / field[0, :, :].max() - sample) ** 2).sum()
    # loss += ((field[32, :, :] / field[16, :, :].max() - sample2) ** 2).sum()
    # loss += ((field[63, :, :] / field[32, :, :].max() - sample3) ** 2).sum()
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"{loss.item():.2f}")

duration = time.time() - start
print(f"Total optimization time for {iters} iterations: ", duration)

field = sim.calculate_volume()
plt.imshow(field[0, :, :].cpu().detach().numpy())
plt.show(block=True)

plt.imshow(field[32, :, :].cpu().detach().numpy())
plt.show(block=True)

plt.imshow(field[63, :, :].cpu().detach().numpy())
plt.show(block=True)

print(sim.transducers[0].amps[:, 0, 0])

viewer, layers = napari.imshow(
    torch.abs(field).rot90(-1, (0, 1)).cpu().detach().numpy()
)
napari.run()
