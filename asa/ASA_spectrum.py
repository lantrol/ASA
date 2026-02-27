import math
import time
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F
from torch.fx.node import Target

# Coordinate System:
# - X: Horizontal axis
# - Y: Depth axis
# - Z: Vertical axis
#
# WARNING: Volume indexes are later defined as [Z, Y, X]
# - This is since the FFT convolution requires the paralel plane
#   to be in the last dimensions


# --- Parameters ---:
#   fr: Frequency of emitters
#   c: speed of sound
#   size: size of volume in meters
#   pos: position of volume in 3D space
#   ds: sice of each cell of the volume
#   device: which torch device to simulate in
#
# Inferred ->
#   dim: size in cells, (size / ds)
#   wavelen: (c / fr)
#   k: (2 * pi / wavelen)
#
# --- Transducer and Propagator coordinates ---
# Transducers are defined as an amp and phase tensors (t_mux, num_emitter, num_emitter).
# The have gradients enabled to allow optimizing them.
# Each iteration, the field is expanded to the simulation dim to allow
# correctly convoluting with FFT. (check to_complex_plane)
#
#
# The propagator is a (dim, dim, dim) tensor. Each transducer/t_mux has each own
# propagator.
# Propagators are always defined vertically respect to each transducer. Later, depending
# on orientation, the propagated volume is rotated and added to the total volume.
# Since Z is the vertical axis, its used to determine the distance from the volume.
#
class Simulation_ASA_Spectrum:
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
            Transducer_Spectrum
        ] = []  # Array of num_emitter x num_emitter phases
        self.propagators: list[torch.Tensor] = []  # Propagator matrix

    def add_transducer(self, transducer):
        same_axis = False
        for tr in self.transducers:
            same_axis = same_axis or tr.orientation == transducer.orientation

        assert not same_axis, "Included diffirent transducers on same axis!"
        self.transducers.append(transducer)

    def emmiter_field_from_phases(self, emitter_dim: int, phases: torch.Tensor):
        assert phases.numel() == emitter_dim * emitter_dim, (
            "Not enough phases for all emitters"
        )

        cell_size = self.size / self.dim
        transducer = torch.zeros(
            (self.dim, self.dim), dtype=torch.complex64, device=self.device
        )

        cells_per_emitter = np.round(self.emitter_size / cell_size).astype(np.int32)
        gap_between = (self.dim - cells_per_emitter * emitter_dim) // emitter_dim

        for y in range(emitter_dim):
            for x in range(emitter_dim):
                pos_x = int(gap_between // 2 + x * (cells_per_emitter + gap_between))
                pos_y = int(gap_between // 2 + y * (cells_per_emitter + gap_between))

                transducer[
                    pos_y : pos_y + cells_per_emitter, pos_x : pos_x + cells_per_emitter
                ] += 1 * torch.exp(1j * phases[y, x])

        return transducer

    def create_propagators(self):
        for transducer in self.transducers:
            transducer.init_transducer(self.ds)
            min_dist = abs(transducer.pos[0])

            fx = torch.fft.fftfreq(self.dim * 2, d=self.ds, device=self.device)
            fy = torch.fft.fftfreq(self.dim * 2, d=self.ds, device=self.device)

            kx = 2 * math.pi * fx
            ky = 2 * math.pi * fy
            kx, ky = torch.meshgrid(kx, ky, indexing="ij")

            kz_squared = self.k**2 - kx**2 - ky**2
            kz = torch.sqrt(kz_squared + 0j)

            dists = torch.linspace(
                min_dist, min_dist + self.ds * self.dim, self.dim, device=self.device
            ).view(-1, 1, 1)

            # Propagation factor
            propagator = torch.exp(1j * kz * (dists + 0j))

            # padded_emitter = pad_func(U0, (pad, pad, pad, pad))

            self.propagators.append(propagator)

    def calculate_volume_spectrum(self):
        volume = torch.zeros(
            (self.transducers[0].t_mux, self.dim, self.dim, self.dim),
            dtype=torch.complex64,
            device=self.device,
        )

        total_mux = self.transducers[0].t_mux
        for idx, tr in enumerate(self.transducers):
            emitter_field = tr.rounded_emitters(self.ds)
            (x, y, z) = emitter_field.size()
            emitter_field = emitter_field.reshape(x, 1, y, z)

            # print(tr.mask.shape)

            # plt.imshow(tr.mask.cpu().detach().numpy())
            # plt.show(block=True)

            # plt.imshow(emitter_field[0, 0, :, :].abs().cpu().detach().numpy())
            # plt.show(block=True)

            # print(emitter_field.shape)

            propagator = self.propagators[idx]

            # plt.imshow(propagator[0, :, :].abs().cpu().detach().numpy())
            # plt.show(block=True)

            pad = emitter_field.shape[-1] // 2
            padded_emitter = F.pad(emitter_field, (pad, pad, pad, pad))

            # print(padded_emitter.shape)

            U0_fft = fft.fft2(padded_emitter)
            Uz_fft = U0_fft * propagator

            Uz = fft.ifft2(Uz_fft)

            # Trim the field
            extra = Uz.size()[-1] - self.dim
            Uz = Uz[
                :,
                :,
                extra // 2 : extra // 2 + self.dim,
                extra // 2 : extra // 2 + self.dim,
            ]

            if tr.orientation == Orientation.Z:
                # for mux in range(transducer.t_mux):
                volume += Uz
            elif tr.orientation == Orientation.Z_1:
                # for mux in range(transducer.t_mux):
                volume += Uz.rot90(2, (1, 2))
            elif tr.orientation == Orientation.Y:
                # for mux in range(transducer.t_mux):
                volume += Uz.rot90(-1, (1, 2))
            elif tr.orientation == Orientation.X:
                # for mux in range(transducer.t_mux):
                volume += Uz.rot90(-1, (1, 3))

        return volume.abs().sum(dim=0) / total_mux

    def calculate_volume(self):
        # For now assume there is only one for testing :P
        volume = torch.zeros(
            (self.transducers[0].t_mux, self.dim, self.dim, self.dim),
            dtype=torch.complex64,
            device=self.device,
        )

        total_mux = 0
        idx = 0
        for transducer in self.transducers:
            total_mux += transducer.t_mux

            emitter = transducer.rounded_emitters(self.ds)

            # plt.imshow(emitter[0, :, :].abs().cpu().detach().numpy())
            # plt.show(block=True)

            (x, y, z) = emitter.size()
            emitter = emitter.reshape(x, 1, y, z)

            propagator = self.propagators[idx]
            idx += 1

            # plt.imshow(propagator[0, :, :].abs().cpu().detach().numpy())
            # plt.show(block=True)

            min_side_size = propagator.shape[-1] + emitter.shape[-1] - 1
            pad_to_size = 2 ** np.ceil(np.log2(min_side_size))

            pad_func = torch.nn.functional.pad

            prop_pad = int(np.ceil((pad_to_size - propagator.shape[-1]) / 2))
            padded_propagator = pad_func(
                propagator, (prop_pad, prop_pad, prop_pad, prop_pad, 0, 0)
            )

            emitt_pad = int(np.ceil((pad_to_size - emitter.shape[-1]) / 2))
            padded_emitter = pad_func(
                emitter, (emitt_pad, emitt_pad, emitt_pad, emitt_pad)
            )

            fft_emitter = torch.fft.fft2(torch.fft.ifftshift(padded_emitter))
            fft_prop = torch.fft.fft2(torch.fft.ifftshift(padded_propagator))

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

            if transducer.orientation == Orientation.Z:
                # for mux in range(transducer.t_mux):
                volume += field
            elif transducer.orientation == Orientation.Z_1:
                # for mux in range(transducer.t_mux):
                volume += field.rot90(2, (1, 2))
            elif transducer.orientation == Orientation.Y:
                # for mux in range(transducer.t_mux):
                volume += field.rot90(-1, (1, 2))
            elif transducer.orientation == Orientation.X:
                # for mux in range(transducer.t_mux):
                volume += field.rot90(-1, (1, 3))

        return volume.abs().sum(dim=0) / total_mux


class Orientation(Enum):
    X = 1
    Y = 2
    Z = 3
    Z_1 = 4


class Transducer_Spectrum:
    def __init__(
        self,
        emitters_num: int,
        array_size: float,
        emitter_size: float,
        pos: list,
        orientation: Orientation = Orientation.Z,
        t_mux: int = 1,
        device="cpu",
    ):
        self.phases: torch.Tensor
        self.amps: torch.Tensor
        self.emitter_size: float = emitter_size
        self.array_size: float = array_size
        self.pos: torch.Tensor = torch.tensor(pos, dtype=torch.float32, device=device)
        self.orientation: Orientation = orientation
        self.t_mux: int = t_mux
        self.device = device

        self.phases = torch.rand(
            (t_mux, emitters_num, emitters_num),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )

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

        transducer = torch.nn.functional.pad(
            field, (pad_left, pad_right, pad_left, pad_right)
        )

        transducer = torch.roll(
            transducer, (gap_between // 2, gap_between // 2), dims=(1, 2)
        )

        return transducer

    def rounded_emitters(self, ds: float):
        target_size = round(self.array_size / ds)

        N = target_size // self.phases.shape[-1]

        complex_field = self.amps * torch.exp(1j * self.phases * 2 * math.pi)
        complex_field = (
            complex_field.repeat_interleave(N, dim=1).repeat_interleave(N, dim=2)
            * self.mask
        )

        return complex_field


def parse_file(path: str = "emitter_vals/phases.txt"):
    data = np.loadtxt(path)
    print(data)
    return data
