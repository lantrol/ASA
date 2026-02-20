from ASA import SimulationASA, Transducer
import numpy as np
import torch
import matplotlib.pyplot as plt
import napari
import time
import os

fr = 40e3
c = 343.0

# t = Transducer(16, 0.16, 0.009, [0, 0, 0])
# matrix = t.to_complex_plane(0.16/128)

# print(matrix)

# plt.imshow(torch.real(matrix).detach().numpy(), vmin=0, vmax=1)
# plt.show(block = True)


sim = SimulationASA(fr, c, ds = 0.16/128)
sim.add_transducer(
    Transducer(8, 0.16, 0.005, [0, 0, 0])
)
sim.create_propagators()
field = sim.calculate_volume()

# plt.imshow(torch.abs(field[0, :, :]).detach().numpy())
# plt.show(block=True)

def trim_field(field):
    final_dim = field.size()[0]
    extra = field.size()[1] - final_dim

    return field[:, extra//2:extra//2 + final_dim, extra//2:extra//2 + final_dim]

field = trim_field(field)
field += field.rot90(2, (0, 1)) # + field.rot90(1, (0, 2))

viewer, layers = napari.imshow(torch.abs(field).rot90(0, (0, 1)).detach().numpy())
napari.run()


exit(-1)

# -- Actual simulation --

sim = SimulationASA(fr, c )
sim.create_propagators()

start = time.time()

sim.add_transducer(16)
field = sim.calculate_volume()

duration = time.time()-start
print("Duration: ", duration)

plt.imshow(torch.abs(field[0, :, :]).detach().numpy())
plt.show(block=True)

viewer, layers = napari.imshow(torch.abs(field).detach().numpy())
napari.run()
