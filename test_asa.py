from ASA import SimulationASA
import numpy as np
import torch
import matplotlib.pyplot as plt
import napari
import time

fr = 40e3
c = 343.0

sim = SimulationASA(fr, c, emitter_size=0.005, dim=128)
sim.create_propagators()

start = time.time()

sim.add_transducer(16)
field = sim.calculate_volume()

duration = time.time()-start
print("Duration: ", duration)

plt.imshow(torch.abs(field[0, :, :]).detach().numpy())
plt.show(block=True)

viewer, layers = napari.imshow(torch.abs(field[:, sim.dim//2:sim.dim//2 + sim.dim, sim.dim//2:sim.dim//2 + sim.dim]).detach().numpy())
napari.run()
