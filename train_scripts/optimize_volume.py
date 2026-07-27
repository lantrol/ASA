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
    column,
    create_donut,
    create_donut_rot,
    cube_frame,
    diagonal_slice,
    helix,
    hollow_sphere,
    sphere,
    vowl,
)

SAVE_OUTPUT = False
LOG_TRAIN = False

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
    round_emitters=False,
):
    ds = 0.16 / sim_dim

    sim = Simulation(fr, c, size=0.16, ds=ds, device=device)

    sim.add_transducer(
        Transducer(
            emitters_num=num_emitters,
            array_size=0.16,
            emitter_size=0.009,
            apperture=0.01,
            pos=[pos_z, 0, 0],
            orientation=Orientation.Z,
            t_mux=t_mux,
            device=device,
            random_init=random_init,
            checkerboard=checkerboard,
            round_emitters=round_emitters,
        )
    )

    if num_arrays > 1:
        sim.add_transducer(
            Transducer(
                emitters_num=num_emitters,
                array_size=0.16,
                emitter_size=0.008,
                apperture=0.01,
                pos=[pos_z, 0, 0],
                orientation=Orientation.Y,
                t_mux=t_mux,
                device=device,
                random_init=random_init,
                checkerboard=checkerboard,
                round_emitters=round_emitters,
            )
        )
    if num_arrays > 2:
        sim.add_transducer(
            Transducer(
                emitters_num=num_emitters,
                array_size=0.16,
                emitter_size=0.008,
                apperture=0.01,
                pos=[pos_z, 0, 0],
                orientation=Orientation.X,
                t_mux=t_mux,
                device=device,
                random_init=random_init,
                checkerboard=checkerboard,
                round_emitters=round_emitters,
            )
        )

    if num_arrays == -1:
        sim.add_transducer(
            Transducer(
                emitters_num=num_emitters,
                array_size=0.16,
                emitter_size=0.008,
                apperture=0.01,
                pos=[pos_z, 0, 0],
                orientation=Orientation.Z_1,
                t_mux=t_mux,
                device=device,
                random_init=random_init,
                checkerboard=checkerboard,
                round_emitters=round_emitters,
            )
        )

    sim.create_propagators()

    # field_sample = sim.transducers[0].to_rounded_emitters(sim.ds)
    # plt.imshow(field_sample.abs()[0, :, :].cpu().detach().numpy())
    # plt.show()

    return sim


def optimize(
    sim,
    target,
    optimizer,
    loss_func,
    iters=800,
    scheduler=None,
    use_mean=False,
):
    print(f"Configuration:")
    # print(f"Loss function: {loss_func.__name__}")
    print(f"Optimizer: {type(optimizer)}")
    if scheduler is not None:
        print(f"Scheduler: {type(scheduler)}")

    start = time.time()
    losses = []
    for k in (pbar := tqdm(range(iters))):
        field = sim.calculate_volume(use_mean=use_mean)

        # loss = loss_func((field / field.max()).flatten(), target.flatten())

        loss = loss_func(field, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(sim.get_params(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        pbar.set_description(f"{loss.item():.5f}")

    duration = time.time() - start
    print(f"Optimization took: {duration:.3f} seconds")
    return losses


def optimize_diagonal(
    sim,
    target,
    optimizer,
    loss_func,
    iters=800,
    scheduler=None,
    use_mean=False,
):
    print(f"Configuration:")
    # print(f"Loss function: {loss_func.__name__}")
    print(f"Optimizer: {type(optimizer)}")
    if scheduler is not None:
        print(f"Scheduler: {type(scheduler)}")

    start = time.time()
    losses = []
    for k in (pbar := tqdm(range(iters))):
        field = sim.calculate_volume(use_mean=use_mean)

        # loss = loss_func((field / field.max()).flatten(), target.flatten())

        # diagonal = torch.zeros((64, 64), dtype=torch.float32, device="cuda")
        # for x in range(64):
        #     for y in range(64):
        #         diagonal[y, x] = field[
        #             y,
        #             y,
        #             x,
        #         ]

        idx = torch.arange(64, device=field.device)
        diagonal = field[idx[:, None], idx[:, None], idx[None, :]]

        loss = loss_func(diagonal, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(sim.get_params(), max_norm=1.0)

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
    loss = (((field / field.max()) - target) ** 2).sum() / field.numel()
    return loss


def mean_absolute_error(field, target):
    loss = (((field / field.max()) - target).abs()).sum() / field.numel()
    return loss


def loss_slice(field, slice, loss_func, layer):
    loss = loss_func(field[layer, :, :], slice)
    return loss


def mean_absolute_percentage_error(field, target):
    loss = (((field / field.max()) - target).abs().sum() * 100) / field.numel()
    return loss


def soft_dice_loss(field, target):
    norm_field = field / field.max()
    return (
        torch.dot(norm_field.flatten(), target.flatten())
        * 2
        / (norm_field.norm() + target.norm())
    )


def main():
    DIM = 64
    NUM_EMITTERS = 64

    # target = torch.roll(create_donut_rot(DIM, 3, 22, [45, 45, 45]), 0, 0)
    # target = (
    #     column(64, 8, 8, 2)
    #     + column(64, 8, 30, 2)
    #     + column(64, 8, 56, 2)
    #     + column(64, 32, 8, 2)
    # )

    image1 = Image.open("./samples/birb.png").convert("L").resize((DIM, DIM))
    sample1 = torch.tensor(np.asarray(image1)[:, :], device="cuda")
    sample1 = sample1 / sample1.max()

    print(sample1.shape)

    # target_img = torch.zeros((DIM, DIM), dtype=torch.float32, device="cuda")
    # for x in range(DIM):
    #     for y in range(DIM):
    #         target_img[y, x] = (
    #             sample1[int(y * np.sqrt(2)), x] if int(y * np.sqrt(2)) < DIM else 0
    #         )

    # target_img = target_img.roll(15, dims=0)

    # target = diagonal_slice(64, target_img)
    target = sphere(dim=DIM, r=0.01).roll([16], dims=[2])
    target += sphere(dim=DIM, r=0.01).roll([-16], dims=[2])

    target = target.to("cuda").type(torch.float32)

    # ---- Array amount optim ----

    ITERS = 2000

    array_nums = [1]

    viewer = napari.Viewer()

    for num_arrays in array_nums:
        sim = init_simulation(
            fr=40e3,
            c=343.0,
            t_mux=4,
            num_emitters=NUM_EMITTERS,
            sim_dim=DIM,
            num_arrays=num_arrays,
            pos_z=-0.08,
            random_init=True,
            checkerboard=False,
        )

        optimizer = torch.optim.Adam(sim.get_params(), 0.1)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=ITERS // 3,
            eta_min=0.001,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            total_steps=ITERS // 2,
        )

        scheduler = None

        losses = optimize(
            sim=sim,
            target=target,
            optimizer=optimizer,
            loss_func=cosine_similarity,
            scheduler=scheduler,
            iters=ITERS,
            use_mean=True,
        )

        # losses = optimize_diagonal(
        #     sim=sim,
        #     target=target_img,
        #     optimizer=optimizer,
        #     loss_func=cosine_similarity,
        #     scheduler=scheduler,
        #     iters=ITERS,
        #     use_mean=True,
        # )

        # plt.plot(losses)
        # plt.ylim(0, 1)
        # plt.show(block=True)

        # with torch.no_grad():
        #     sim.transducers[0].amps[:, :14, :8] = -5

        # sim.transducers[0].phases = torch.zeros(
        #     (4, 16, 16), dtype=torch.float32, requires_grad=True, device="cuda"
        # )
        field = sim.calculate_volume(use_mean=True)

        viewer.add_image(
            torch.abs(field).rot90(-0, (0, 1)).cpu().detach().numpy(),
            name=f"{num_arrays}",
        )

        viewer.add_image(
            target.rot90(-0, (0, 1)).cpu().detach().numpy(),
            name=f"Target",
        )

        slice = torch.zeros((64, 64), dtype=torch.float32)
        for x in range(64):
            for y in range(64):
                slice[y, x] = field[
                    y,
                    y,
                    x,
                ]

        viewer.add_image(
            slice.rot90(-0, (0, 1)).cpu().detach().numpy(),
            name=f"Slice",
        )

        if LOG_TRAIN:
            print(f"With {num_arrays} arrays -> {losses[-1]}")

        if SAVE_OUTPUT:
            field = sim.calculate_volume()

            phases = sim.transducers[0].phases
            print(phases.shape)
            print(phases)

            np.savetxt(
                f"emitter_vals/all_on.txt",
                phases.reshape(4, 16 * 16).cpu().detach().numpy(),
            )

    napari.run()


if __name__ == "__main__":
    main()
