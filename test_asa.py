import math
import os
import time

import matplotlib.pyplot as plt
import napari
import numpy as np
import torch

from ASA import Orientation, SimulationASA, Transducer

fr = 40e3
c = 343.0

sim = SimulationASA(fr, c, size=0.08, ds=0.08 / 64)
sim.add_transducer(Transducer(16, 0.16, 0.005, [0, 0, 0], Orientation.Z))
sim.add_transducer(Transducer(16, 0.16, 0.005, [0, 0, 0], Orientation.X))
sim.add_transducer(Transducer(16, 0.16, 0.005, [0, 0, 0], Orientation.Y))
sim.create_propagators()


size = 64
sigma = size / 24  # controls spread (smaller = sharper sphere)

# Create coordinate grid
coords = torch.linspace(-1, 1, size)
x, y, z = torch.meshgrid(coords, coords, coords, indexing="ij")

# Squared distance from center
r2 = x**2 + y**2 + z**2

# Gaussian
sample = torch.exp(-r2 / (2 * (sigma / size) ** 2)) * 10

print(sample.mean())

field = None
for k in range(20):
    field = sim.calculate_volume()

    loss = (field.abs() - sample).abs().sum()
    loss.backward()

    print(loss.item())

    with torch.no_grad():
        for tr in sim.transducers:
            tr.phases -= 0.001 * tr.phases.grad
            tr.amps -= 0.001 * tr.amps.grad

    for tr in sim.transducers:
        tr.phases.grad = None
        tr.amps.grad = None

field = sim.calculate_volume()
viewer, layers = napari.imshow(torch.abs(field).rot90(-1, (0, 1)).detach().numpy())
napari.run()
