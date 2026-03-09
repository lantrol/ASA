import math
import os
import time

import glm
import matplotlib.pyplot as plt
import napari
import numpy as np
import torch
from PIL import Image
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
SKIP_TRAINING = False

torch.manual_seed(68)


def init_simulation(
    t_mux=16,
    fr=40e3,
    c=343.0,
    sim_dim=64,
    num_arrays: int = 3,
    pos_z=-0.11,
    device="cuda",
    random_init=True,
):
    ds = 0.16 / sim_dim

    sim = Simulation_Bruteforce(fr, c, size=0.16, ds=ds, device=device)
    print(sim.dim)

    sim.add_transducer(
        Transducer_Bruteforce(
            emitters_num=16,
            array_size=0.16,
            emitter_size=ds,
            apperture=0.01,
            pos=[pos_z, 0, 0],
            orientation=Orientation_Bruteforce.Z,
            t_mux=t_mux,
            device=device,
            random_init=random_init,
        )
    )

    if num_arrays > 1:
        sim.add_transducer(
            Transducer_Bruteforce(
                emitters_num=16,
                array_size=0.16,
                emitter_size=ds,
                apperture=0.01,
                pos=[pos_z, 0, 0],
                orientation=Orientation_Bruteforce.Y,
                t_mux=t_mux,
                device=device,
                random_init=random_init,
            )
        )
    if num_arrays > 2:
        sim.add_transducer(
            Transducer_Bruteforce(
                emitters_num=16,
                array_size=0.16,
                emitter_size=ds,
                apperture=0.01,
                pos=[pos_z, 0, 0],
                orientation=Orientation_Bruteforce.X,
                t_mux=t_mux,
                device=device,
                random_init=random_init,
            )
        )

    sim.create_propagators()

    return sim


def optimize_slices(sim, targets, optimizer, loss_func, iters=800, scheduler=None):
    print(f"Configuration:")
    print(f"Loss function: {loss_func.__name__}")
    print(f"Optimizer: {type(optimizer)}")
    if scheduler is not None:
        print(f"Scheduler: {type(scheduler)}")

    start = time.time()
    losses = []
    for k in (pbar := tqdm(range(iters))):
        field = sim.calculate_volume()

        loss = loss_slice(field, targets[0], loss_func, 0)
        loss += loss_slice(field, targets[1], loss_func, 32)
        loss += loss_slice(field, targets[2], loss_func, 63)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        pbar.set_description(f"{loss.item():.5f}")

    duration = time.time() - start
    print(f"Optimization took: {duration:.3f} seconds")
    return losses


def optimize(sim, target, optimizer, loss_func, iters=800, scheduler=None):
    print(f"Configuration:")
    print(f"Loss function: {loss_func.__name__}")
    print(f"Optimizer: {type(optimizer)}")
    if scheduler is not None:
        print(f"Scheduler: {type(scheduler)}")

    start = time.time()
    losses = []
    for k in (pbar := tqdm(range(iters))):
        field = sim.calculate_volume()

        loss = loss_func(field, target)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        pbar.set_description(f"{loss.item():.5f}")

    duration = time.time() - start
    print(f"Optimization took: {duration:.3f} seconds")
    return losses


def cosine_similarity(field, target):
    loss = 1 - torch.dot(field.flatten(), target.flatten()).sum() ** 2 / (
        (field**2).sum() * (target**2).sum() + 1e-8
    )

    return loss


def mean_squared_error(field, target):
    loss = ((field / field.max() - target) ** 2).sum() / field.numel()
    return loss


def mean_absolute_error(field, target):
    loss = ((field / field.max() - target).abs()).sum() / field.numel()
    return loss


def loss_slice(field, slice, loss_func, layer):
    loss = loss_func(field[layer, :, :], slice)
    return loss


def main():
    sim = init_simulation(
        t_mux=16,
        fr=40e3,
        c=343.0,
        sim_dim=64,
        num_arrays=3,
        pos_z=-0.12,
        random_init=True,
    )

    target = torch.roll(create_donut_rot(sim.dim, 4, 20, [45, 45, 45]), 0, 0)
    target = target.to("cuda").type(torch.float32)

    image = Image.open("./samples/domino.png").convert("L")
    sample = torch.tensor(np.asarray(image)[:, :], device="cuda")
    sample1 = sample / sample.max()

    image2 = Image.open("./samples/smiley.png").convert("L")
    sample2 = torch.tensor(np.asarray(image2)[::2, ::2], device="cuda")
    sample2 = sample2 / sample2.max()

    image3 = Image.open("./samples/loss.png").convert("L")
    sample3 = torch.tensor(np.asarray(image3)[:, :], device="cuda")
    sample3 = sample3 / sample3.max()

    samples = [sample1, sample2, sample3]

    loss_functions = [cosine_similarity]  # , mean_absolute_error, mean_squared_error]

    viewer = napari.Viewer()

    for func in loss_functions:
        for tr in sim.transducers:
            tr.reset_params()

        optimizer = torch.optim.Adam(sim.get_params(), 0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [800, 1200], gamma=0.2
        )
        # scheduler = None

        optimize(
            sim=sim,
            target=target,
            optimizer=optimizer,
            loss_func=func,
            scheduler=scheduler,
            iters=1500,
        )

        field = sim.calculate_volume()
        # field_brute = sim.calculate_volume_brute()

        viewer.add_image(
            torch.abs(field).rot90(-0, (0, 1)).cpu().detach().numpy(),
            name=func.__name__,
        )
        # viewer.add_image(
        #     torch.abs(field_brute).rot90(-0, (0, 1)).cpu().detach().numpy(),
        #     name="brute",
        # )

        if SAVE_OUTPUT:
            tr0 = sim.transducers[0]
            tr1 = sim.transducers[1]
            tr2 = sim.transducers[2]

            all_amps = torch.stack([tr0.amps, tr1.amps, tr2.amps])
            all_phases = torch.stack([tr0.phases, tr1.phases, tr2.phases])

            torch.save(all_amps, "cross_compare/all_amps.pt")
            torch.save(all_phases, "cross_compare/all_phases.pt")
            torch.save(field, "cross_compare/volume.pt")

    napari.run()


if __name__ == "__main__":
    main()
