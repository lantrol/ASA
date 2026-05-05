import itertools
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import napari
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from asa import (
    Orientation,
    Simulation,
    Transducer,
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
LOG_TRAIN = True

torch.manual_seed(68)


def init_simulation(
    fr=40e3,
    c=343.0,
    t_mux=16,
    num_emitters=16,
    sim_dim=64,
    num_arrays: int = 3,
    pos_z=-0.11,
    device="cuda",
    random_init=True,
    checkerboard=False,
    optimize_amps=True,
):
    ds = 0.16 / sim_dim

    sim = Simulation(
        fr, c, size=0.16, ds=ds, device=device, optimize_amps=optimize_amps
    )

    transducer = Transducer(
        emitters_num=num_emitters,
        array_size=0.16,
        emitter_size=ds,
        apperture=0.01,
        pos=[pos_z, 0, 0],
        orientation=Orientation.Z,
        t_mux=t_mux,
        device=device,
        random_init=random_init,
        checkerboard=checkerboard,
    )

    if not optimize_amps:
        transducer.amps = torch.ones(
            (t_mux, num_emitters, num_emitters),
            dtype=torch.float32,
            requires_grad=True,
            device=device,
        )

        with torch.no_grad():
            transducer.amps *= 5

    sim.add_transducer(transducer)

    if num_arrays == -1:
        sim.add_transducer(
            Transducer(
                emitters_num=num_emitters,
                array_size=0.16,
                emitter_size=ds,
                apperture=0.01,
                pos=[pos_z, 0, 0],
                orientation=Orientation.Z_1,
                t_mux=t_mux,
                device=device,
                random_init=random_init,
                checkerboard=checkerboard,
            )
        )

    sim.create_propagator_slices(0)

    field_sample = sim.transducers[0].to_rounded_emitters(sim.ds)

    plt.imshow(field_sample.abs()[0, :, :].cpu().detach().numpy())
    plt.show()

    plt.imshow(sim.slices[0].abs()[0, :, :].cpu().detach().numpy())
    plt.show()

    return sim


def optimize_slices(
    sim, targets, optimizer, loss_func, iters=800, scheduler=None, use_mean=False
):
    assert len(targets) > 0, "No target slices passed"

    print(f"Configuration:")
    print(f"Loss function: {loss_func.__name__}")
    print(f"Optimizer: {type(optimizer)}")
    if scheduler is not None:
        print(f"Scheduler: {type(scheduler)}")

    start = time.time()
    losses = []
    vals = []
    for k in (pbar := tqdm(range(iters))):
        field = sim.calculate_slices(use_mean=use_mean)

        loss = loss_slice(field, targets[0], loss_func, 0)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        vals.append(optimizer.param_groups[0]["lr"])
        pbar.set_description(f"{loss.item():.5f}")

    duration = time.time() - start
    print(f"Optimization took: {duration:.3f} seconds")
    # plt.plot(vals)
    # plt.show(block=True)
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
    field_norm = field / field.max()
    loss = 1 - torch.dot(field_norm.flatten(), target.flatten()).sum() ** 2 / (
        (field_norm**2).sum() * (target**2).sum() + 1e-8
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
    image1 = Image.open("./samples/birb_small.png").convert("L")
    sample1 = torch.tensor(np.asarray(image1)[:, :], device="cuda")
    sample1 = sample1 / sample1.max()

    image2 = Image.open("./samples/smiley.png").convert("L")
    sample2 = torch.tensor(np.asarray(image2)[::2, ::2], device="cuda")
    sample2 = sample2 / sample2.max()

    image3 = Image.open("./samples/loss.png").convert("L")
    sample3 = torch.tensor(np.asarray(image3)[:, :], device="cuda")
    sample3 = sample3 / sample3.max()

    image4 = Image.open("./samples/domino.png").convert("L")
    sample4 = torch.tensor(np.asarray(image4)[:, :], device="cuda")
    sample4 = sample4 / sample4.max()

    samples = [
        sample2,
    ]

    # ---- Array amount optim ----

    ITERS = 400

    arrays = [1]

    for num_arrays in arrays:
        sim = init_simulation(
            fr=40e3,
            c=343.0,
            sim_dim=64,
            t_mux=4,
            num_emitters=16,
            num_arrays=num_arrays,
            pos_z=-0.16,
            random_init=True,
            checkerboard=False,
            optimize_amps=False,
        )

        optimizer = torch.optim.Adam(sim.get_params(), 0.01)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, ITERS, eta_min=0.001
        # )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            total_steps=ITERS,
        )

        losses = optimize_slices(
            sim=sim,
            targets=samples,
            optimizer=optimizer,
            loss_func=cosine_similarity,
            scheduler=scheduler,
            iters=ITERS,
            use_mean=True,
        )

        field = sim.calculate_slices(use_mean=True)

        viewer = napari.Viewer()
        viewer.add_image(
            torch.abs(field).cpu().detach().numpy(),
            name=f"{num_arrays}",
        )

        phases = sim.transducers[0].phases
        phases = phases.reshape(4, 16 * 16)
        np.savetxt(
            f"emitter_vals/drawing_phases_attenuation.txt",
            phases.cpu().detach().numpy(),
        )

        # amps = sim.transducers[0].amps
        # print(amps.min(), amps.max())

        if LOG_TRAIN:
            print(f"With {num_arrays} arrays -> {losses[-1]}")

        if SAVE_OUTPUT:
            field = sim.calculate_volume()

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
