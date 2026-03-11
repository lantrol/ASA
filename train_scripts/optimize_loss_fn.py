import itertools
import json
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
LOG_TRAIN = True

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

    target = torch.roll(create_donut_rot(sim.dim, 4, 20, [0, 0, 0]), 0, 0)
    target = target.to("cuda").type(torch.float32)

    # ---- Optimization parameters and loop ----

    ITERS = 1500
    utc_time = datetime.now(timezone.utc)

    learning_rates = [0.1, 0.03, 0.01, 0.001, 0.0001]

    loss_functions = [
        cosine_similarity,
        mean_squared_error,
        mean_absolute_error,
    ]

    schedulers = [
        (None, {}),
        (torch.optim.lr_scheduler.CosineAnnealingLR, {"T_max": ITERS}),
        (
            torch.optim.lr_scheduler.CosineAnnealingLR,
            {"T_max": ITERS // 3, "eta_min": 0.001},
        ),
        (
            torch.optim.lr_scheduler.MultiStepLR,
            {"milestones": [800, 1200], "gamma": 0.2},
        ),
    ]

    all_configs = list(itertools.product(learning_rates, loss_functions, schedulers))

    for i, config in enumerate(all_configs):
        print(f"\n --- Config {i + 1} of {len(all_configs)} ---")

        learning_rate = config[0]
        loss_func = config[1]
        scheduler_conf = config[2]

        for tr in sim.transducers:
            tr.reset_params()

        optimizer = torch.optim.Adam(sim.get_params(), learning_rate)

        scheduler = None
        if scheduler_conf[0] is not None:
            scheduler = scheduler_conf[0](optimizer, **scheduler_conf[1])

        losses = optimize(
            sim=sim,
            target=target,
            optimizer=optimizer,
            loss_func=loss_func,
            scheduler=scheduler,
            iters=ITERS,
        )

        if LOG_TRAIN:
            info = {
                "base_lr": learning_rate,
                "loss_function": loss_func.__name__,
                "scheduler": {
                    "scheduler_name": "None"
                    if scheduler_conf[0] is None
                    else scheduler_conf[0].__name__,
                    "scheduler_conf": scheduler_conf[1],
                },
                "losses": losses,
            }

            with open(f"train_runs/{utc_time}-config-{i + 1}.json", "w") as json_file:
                json.dump(info, json_file, indent=4)

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


if __name__ == "__main__":
    main()
