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

# sim.add_transducer(
#     Transducer(16, 0.16, 0.008, [0, 0, 0], Orientation.X, t_mux=1, device=device)
# )

sim.add_transducer(
    Transducer(16, 0.16, 0.008, [-0.12, 0, 0], Orientation.X, t_mux=16, device=device)
)
sim.add_transducer(
    Transducer(16, 0.16, 0.008, [-0.12, 0, 0], Orientation.Y, t_mux=16, device=device)
)
sim.add_transducer(
    Transducer(16, 0.16, 0.008, [-0.12, 0, 0], Orientation.Z, t_mux=16, device=device)
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


params = []

for tr in sim.transducers:
    for t in range(tr.t_mux):
        params.append(tr.phases[t])
        params.append(tr.amps[t])

optimizer = torch.optim.AdamW(params, lr=0.1)
for k in range(300):
    field = sim.calculate_volume()
    field = field.abs()  # / torch.abs(field).max()

    loss = ((field - sample) ** 2).sum()
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"{loss.item():.2f}")

    # with torch.no_grad():
    #     for tr in sim.transducers:
    #         for t in range(tr.t_mux):
    #             tr.phases[t] -= 0.00000001 * tr.phases[t].grad
    #             tr.amps[t] -= 0.00000001 * tr.amps[t].grad

    # for tr in sim.transducers:
    #     for t in range(tr.t_mux):
    #         tr.phases[t].grad = None
    #         tr.amps[t].grad = None

field = sim.calculate_volume()

print(torch.abs(field).mean())

field_norm = field.abs() / torch.abs(field).max()
loss = (field_norm - sample).abs()

viewer, layers = napari.imshow(
    torch.abs(field).rot90(-1, (0, 1)).cpu().detach().numpy()
)
napari.run()
