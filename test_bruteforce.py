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
    Orientation_Bruteforce,
    Simulation_Bruteforce,
    Transducer_Bruteforce,
)
from volume_utils import (
    create_donut,
    create_donut_rot,
    cube_frame,
    helix,
    hollow_sphere,
    vowl,
)

SAVE_OUTPUT = False
SKIP_TRAINING = True

torch.manual_seed(68)

t_mux = 4
fr = 40e3
c = 343.0
wavelen = c / fr
sim_dim = 128
ds = 0.16 / sim_dim

device = "cuda"

sim = Simulation_Bruteforce(fr, c, size=0.16, ds=ds, device=device)
print(sim.dim)


# sim.add_transducer(
#     Transducer_Bruteforce(
#         emitters_num=16,
#         array_size=0.16,
#         emitter_size=ds,
#         apperture=0.01,
#         pos=[-0.12, 0, 0],
#         orientation=Orientation_Bruteforce.X,
#         t_mux=t_mux,
#         device=device,
#     )
# )
# sim.add_transducer(
#     Transducer_Bruteforce(
#         emitters_num=16,
#         array_size=0.16,
#         emitter_size=ds,
#         apperture=0.01,
#         pos=[-0.12, 0, 0],
#         orientation=Orientation_Bruteforce.Y,
#         t_mux=t_mux,
#         device=device,
#     )
# )
sim.add_transducer(
    Transducer_Bruteforce(
        emitters_num=2,
        array_size=0.16,
        emitter_size=ds,
        apperture=0.01,
        pos=[-0.001, 0, 0],
        orientation=Orientation_Bruteforce.Z,
        t_mux=t_mux,
        device=device,
        random_init=False,
    )
)

sim.create_propagators()

plt.imshow(sim.transducers[0].mask.cpu().detach().numpy())
plt.show(block=True)


image = Image.open("./samples/birb.png").convert("L")
sample = torch.tensor(np.asarray(image)[:, :], device=device)
sample = sample / sample.max()

image2 = Image.open("./samples/smiley.png").convert("L")
sample2 = torch.tensor(np.asarray(image2)[::2, ::2], device=device)
sample2 = sample2 / sample2.max()


image3 = Image.open("./samples/loss.png").convert("L")
sample3 = torch.tensor(np.asarray(image3)[:, :], device=device)
sample3 = sample3 / sample3.max()


# sample = hollow_sphere(sim_dim, 0.16, 0.041)
sample = torch.roll(create_donut_rot(sim_dim, 3, 20, [0, 0, 0]), 0, 0)

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

iters = 200

optimizer = torch.optim.Adam(params, lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 300, gamma=0.1)
losses = []
for k in (pbar := tqdm(range(iters))):
    if SKIP_TRAINING:
        break
    field = sim.calculate_volume()
    field = field

    # Volume
    loss = ((field - sample) ** 2).sum() / field.numel()

    # loss = 1 - torch.dot(field.flatten(), sample.flatten()).sum() ** 2 / (
    #     (field**2).sum() * (sample**2).sum() + 1e-8
    # )

    loss.backward()

    # Slice a bit bruteforce-y
    # loss = (torch.abs(field[0, :, :] / field[0, :, :].max() - sample)).sum()
    # loss += (torch.abs(field[32, :, :] / field[32, :, :].max() - sample2)).sum()
    # loss += (torch.abs(field[63, :, :] / field[63, :, :].max() - sample3)).sum()
    # loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    losses.append(loss.item())
    pbar.set_description(f"{losses[-1]:.5f}")


duration = time.time() - start
# print(f"Total optimization time for {iters} iterations: ", duration)
# print(f"Final loss: {losses[-1]:.2f}")

# plt.plot(range(iters), losses)
# plt.ylim((0, 1))
# plt.show(block=True)

field = sim.calculate_volume()

# Saving data to compare
if SAVE_OUTPUT:
    tr0 = sim.transducers[0]
    tr1 = sim.transducers[1]
    tr2 = sim.transducers[2]

    all_amps = torch.stack([tr0.amps, tr1.amps, tr2.amps])
    all_phases = torch.stack([tr0.phases, tr1.phases, tr2.phases])

    torch.save(all_amps, "cross_compare/all_amps.pt")
    torch.save(all_phases, "cross_compare/all_phases.pt")
    torch.save(field, "cross_compare/volume.pt")


print(sim.transducers[0].amps[:, 0, 0])

# field_brute = sim.calculate_volume_brute()

viewer = napari.Viewer()

# viewer.add_image(
#     torch.abs(field_brute).rot90(-0, (0, 1)).cpu().detach().numpy(), name="bruteforce"
# )
viewer.add_image(
    torch.abs(field).rot90(-0, (0, 1)).cpu().detach().numpy(), name="small_conv"
)

# viewer.add_image(diff.rot90(-0, (0, 1)).cpu().detach().numpy())
napari.run()
