import math
import os
import time

import matplotlib.pyplot as plt
import napari
import numpy as np
import torch

from ASA import Orientation, SimulationASA, Transducer

torch.manual_seed(68)

fr = 40e3
c = 343.0

device = "cuda"

sim = SimulationASA(fr, c, size=0.16, ds=0.16 / 25, device=device)


sim.add_transducer(
    Transducer(16, 0.16, 0.008, [-0.12, 0, 0], Orientation.X, t_mux=1, device=device)
)
sim.add_transducer(
    Transducer(16, 0.16, 0.008, [-0.12, 0, 0], Orientation.Y, t_mux=1, device=device)
)
sim.add_transducer(
    Transducer(16, 0.16, 0.008, [-0.12, 0, 0], Orientation.Z, t_mux=1, device=device)
)

sim.create_propagators()


size = 25
sigma = size / 12  # controls spread (smaller = sharper sphere)

# Create coordinate grid
coords = torch.linspace(-1, 1, size)
x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")

# Squared distance from center
r2 = x**2 + y**2 + z**2

# Gaussian
sample = torch.exp(-r2 / (2 * (sigma / size) ** 2)).to(device) * 2

# print(sample.max())


start = time.time()

params = []
for tr in sim.transducers:
    params.append(tr.phases)
    params.append(tr.amps)

iters = 300
optimizer = torch.optim.AdamW(params, lr=0.1)
for k in range(iters):
    field = sim.calculate_volume()
    field = field.abs()  # / torch.abs(field).max()

    loss = ((field - sample) ** 2).sum()
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"{loss.item():.2f}")

duration = time.time() - start
print(f"Total optimization time for {iters} iterations: ", duration)

field = sim.calculate_volume()

viewer, layers = napari.imshow(
    torch.abs(field).rot90(-1, (0, 1)).cpu().detach().numpy()
)
napari.run()
