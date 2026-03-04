import math
import time
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# WIP:
# Trying to see if something similar to ASA Convolution can be used for bruteforce method


class Simulation_Bruteforce:
    def __init__(
        self,
        fr,
        c,
        size=0.16,
        pos: list[float] = [0, 0, 0.09],
        ds=0.16 / 64,
        device="cpu",
    ):
        # Common data for all fields
        self.fr = fr
        self.c = c
        self.wavelen = c / fr
        self.k = 2 * math.pi / self.wavelen
        self.pos = torch.tensor(pos, dtype=torch.float32, device=device)
        self.ds = ds
        self.size = size
        self.dim = int(self.size / self.ds)
        self.device = device

        self.transducers: list[
            Transducer_Bruteforce
        ] = []  # Array of num_emitter x num_emitter phases
        self.propagators: list[torch.Tensor] = []  # Propagator matrix

    def add_transducer(self, transducer):
        same_axis = False
        for tr in self.transducers:
            same_axis = same_axis or tr.orientation == transducer.orientation
        self.transducers.append(transducer)

    def create_propagators(self):
        for transducer in self.transducers:
            transducer.init_transducer(self.ds)

            # TEST: reuse only 1 for memory
            if len(self.propagators) > 0:
                continue

            # TEST: Propagator of needed size from start
            propagator = torch.zeros(
                (self.dim, self.dim * 2, self.dim * 2),
                dtype=torch.complex64,
                device=self.device,
            )

            # Create the coordinate grid
            x_coords = (
                torch.arange(-self.dim, self.dim, device=self.device)
                .view(1, 1, -1)
                .float()
                * self.ds
                + self.pos[0]
            )
            y_coords = (
                torch.arange(-self.dim, self.dim, device=self.device)
                .view(1, -1, 1)
                .float()
                * self.ds
                + self.pos[1]
            )
            z_coords = (
                torch.arange(-self.dim // 2, self.dim // 2, device=self.device)
                .view(-1, 1, 1)
                .float()
                * self.ds
                + self.pos[2]
            )

            # Expand dimensions for broadcasting to match the propagator shape [dim, dim, dim]
            x_coords = x_coords.expand(self.dim, self.dim * 2, -1)
            y_coords = y_coords.expand(self.dim, -1, self.dim * 2)
            z_coords = z_coords.expand(-1, self.dim * 2, self.dim * 2)

            cell_pos = torch.stack(
                [
                    z_coords,
                    y_coords,
                    x_coords,
                ],
                dim=0,
            )

            # Ensure transducer.pos broadcasts to (3, 1, 1, 1)
            transducer_pos_broad = transducer.pos.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            between = cell_pos - transducer_pos_broad
            dist = torch.norm(between, dim=0)

            propagator = (1.0 / (dist + 1e-9)) * torch.exp(1j * (self.k * dist))

            self.propagators.append(propagator)

    def calculate_volume(self):
        # For now assume there is only one for testing :P
        volume = torch.zeros(
            (self.transducers[0].t_mux, self.dim, self.dim, self.dim),
            dtype=torch.complex64,
            device=self.device,
        )

        total_mux = 0
        for idx, transducer in enumerate(self.transducers):
            total_mux += transducer.t_mux

            emitter = transducer.rounded_emitters(self.ds)

            # plt.imshow(emitter[0, :, :].abs().cpu().detach().numpy())
            # plt.show(block=True)

            (x, y, z) = emitter.size()
            emitter = emitter.reshape(x, 1, y, z)

            # TEST: Always use first propagator to save memory (there is only 1)
            propagator = self.propagators[0]

            # plt.imshow(propagator[0, :, :].abs().cpu().detach().numpy())
            # plt.show(block=True)

            # TEST: Only pad emitter, propagator comes in needed size
            # min_side_size = propagator.shape[-1] + emitter.shape[-1] - 1
            # pad_to_size = 2 ** np.ceil(np.log2(min_side_size))

            # prop_pad = int(np.ceil((pad_to_size - propagator.shape[-1]) / 2))
            # padded_propagator = F.pad(
            #     propagator, (prop_pad, prop_pad, prop_pad, prop_pad, 0, 0)
            # )

            # emitt_pad = int(np.ceil((pad_to_size - emitter.shape[-1]) / 2))
            # padded_emitter = F.pad(
            #     emitter, (emitt_pad, emitt_pad, emitt_pad, emitt_pad)
            # )

            padded_emitter = F.pad(
                emitter, (self.dim // 2, self.dim // 2, self.dim // 2, self.dim // 2)
            )

            fft_emitter = torch.fft.fft2(torch.fft.ifftshift(padded_emitter))
            fft_prop = torch.fft.fft2(torch.fft.ifftshift(propagator))

            convolved = fft_emitter * fft_prop

            field = torch.fft.fftshift(torch.fft.ifft2(convolved))

            # Trim the field
            extra = field.size()[-1] - self.dim
            field = field[
                :,
                :,
                extra // 2 : extra // 2 + self.dim,
                extra // 2 : extra // 2 + self.dim,
            ]

            if transducer.orientation == Orientation_Bruteforce.Z:
                # for mux in range(transducer.t_mux):
                volume += field
            elif transducer.orientation == Orientation_Bruteforce.Z_1:
                # for mux in range(transducer.t_mux):
                volume += field.rot90(2, (1, 2))
            elif transducer.orientation == Orientation_Bruteforce.Y:
                # for mux in range(transducer.t_mux):
                volume += field.rot90(-1, (1, 2))
            elif transducer.orientation == Orientation_Bruteforce.X:
                # for mux in range(transducer.t_mux):
                volume += field.rot90(-1, (1, 3))

        return volume.abs().sum(dim=0) / total_mux


class Orientation_Bruteforce(Enum):
    X = 1
    Y = 2
    Z = 3
    Z_1 = 4


class Transducer_Bruteforce:
    def __init__(
        self,
        emitters_num: int,
        array_size: float,
        emitter_size: float,
        pos: list,
        orientation: Orientation_Bruteforce = Orientation_Bruteforce.Z,
        t_mux: int = 1,
        device="cpu",
    ):
        self.phases: torch.Tensor
        self.amps: torch.Tensor
        self.emitter_size: float = emitter_size
        self.array_size: float = array_size
        self.pos: torch.Tensor = torch.tensor(pos, dtype=torch.float32, device=device)
        self.orientation: Orientation_Bruteforce = orientation
        self.t_mux: int = t_mux
        self.device = device

        self.phases = torch.rand(
            (t_mux, emitters_num, emitters_num),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )

        with torch.no_grad():
            self.phases *= 2 * math.pi

        self.amps = torch.rand(
            (t_mux, emitters_num, emitters_num),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )

        # Persistent info por complex plane calculation
        # Used to make calculations faster by saving information

        self.mask: torch.Tensor = None

    def init_transducer(self, ds: float):
        # This creates a mask that will be used later when creating the complex plane
        # Made separately and saved to speed up execution a bit

        target_size = round(self.array_size / ds)

        gap_between = self.array_size / self.phases.shape[-1]

        dim_range_x = torch.arange(
            0, target_size, dtype=torch.float32, device=self.device
        ).view((1, -1))
        dim_range_y = torch.arange(
            0, target_size, dtype=torch.float32, device=self.device
        ).view((-1, 1))

        positions = torch.stack(
            [
                dim_range_x.expand((target_size, -1)),
                dim_range_y.expand((-1, target_size)),
            ],
            dim=0,
        )

        positions = ((positions * ds) % gap_between) - gap_between / 2
        dists = torch.sqrt((positions**2).sum(dim=0))

        self.mask = dists <= self.emitter_size / 2

        # roll_size = int(round((gap_between / 2) / ds))
        # self.mask = torch.roll(self.mask, (-roll_size, -roll_size), (0, 1))

    def to_complex_plane(self, ds: float):
        target_size = round(self.array_size / ds)

        cells_per_emitter = int(np.round(self.emitter_size / ds))
        n_y = self.phases.size(1)

        gap_between = (target_size - cells_per_emitter * n_y) // n_y

        # (t_mux, ny, nx)
        field = self.amps * torch.exp(1j * self.phases)  # complex64

        # Insert gaps using Kronecker trick
        if gap_between > 0:
            gap_kernel = torch.zeros(
                (cells_per_emitter + gap_between, cells_per_emitter + gap_between),
                dtype=torch.complex64,
                device=self.device,
            )
            gap_kernel[:cells_per_emitter, :cells_per_emitter] = 1

            field = torch.kron(field, gap_kernel)

        # Center padding
        pad_total = target_size - field.shape[-1]
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        transducer = F.pad(field, (pad_left, pad_right, pad_left, pad_right))

        transducer = torch.roll(
            transducer, (gap_between // 2, gap_between // 2), dims=(1, 2)
        )

        return transducer

    def rounded_emitters(self, ds: float):
        target_size = round(self.array_size / ds)

        N = target_size // self.phases.shape[-1]

        complex_field = self.amps * torch.exp(1j * self.phases)
        complex_field = (
            complex_field.repeat_interleave(N, dim=1).repeat_interleave(N, dim=2)
            * self.mask
        )

        return complex_field
