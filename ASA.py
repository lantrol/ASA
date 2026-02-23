import math
import time
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

# Idea Dump:
#  - Transducer:
#    Will save a 2D array of Phases and Amps for each emitter
#    Befor calculating, convert to complex field of amps with same
#    resolution as volume slice (have to see how to do correctly)
#
#  - Propagator:
#     Field of complex numbers codifying the distances
#     Have to correctly set center and carefull with transducer positions.
#     If transducers only orthogonal, dont need for precise coordinates
#     (just flip matrixes befor adding to volume)
#

# Coordinate System:
# - X: Horizontal axis
# - Y: Depth axis
# - Z: Vertical axis
#
# WARNING: Volume indexes are later defined as [Z, Y, X]
# - This is since the FFT convolution requires the paralel plane
#   to be in the last dimensions


class SimulationASA:
    # --- Parameters ---:
    # fr: Frequency of emitters
    # c: speed of sound

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
            Transducer
        ] = []  # Array of num_emitter x num_emitter phases
        self.propagator: list[torch.Tensor] = []  # Propagator matrix

    def add_transducer(self, transducer):
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
            propagator = torch.zeros(
                (self.dim, self.dim, self.dim),
                dtype=torch.complex64,
                device=self.device,
            )

            # Create the coordinate grid
            z_coords = (
                torch.arange(-self.dim // 2, self.dim // 2, device=self.device)
                .view(1, 1, -1)
                .float()
                * self.ds
                + self.pos[0]
            )
            y_coords = (
                torch.arange(-self.dim // 2, self.dim // 2, device=self.device)
                .view(1, -1, 1)
                .float()
                * self.ds
                + self.pos[1]
            )
            x_coords = (
                torch.arange(-self.dim // 2, self.dim // 2, device=self.device)
                .view(-1, 1, 1)
                .float()
                * self.ds
                + self.pos[2]
            )

            # Expand dimensions for broadcasting to match the propagator shape [dim, dim, dim]
            z_coords = z_coords.expand(self.dim, self.dim, -1)
            y_coords = y_coords.expand(self.dim, -1, self.dim)
            x_coords = x_coords.expand(-1, self.dim, self.dim)

            cell_pos = torch.stack(
                [
                    x_coords,
                    y_coords,
                    z_coords,
                ],
                dim=0,
            )

            # Ensure transducer.pos broadcasts to (3, 1, 1, 1)
            transducer_pos_broad = transducer.pos.unsqueeze(1).unsqueeze(1).unsqueeze(1)

            between = cell_pos - transducer_pos_broad
            dist = torch.norm(between, dim=0)

            propagator = (1.0 / (dist + 1e-9)) * torch.exp(1j * (self.k * dist))

        self.propagator.append(propagator)

    def calculate_volume(self):
        # For now assume there is only one for testing :P
        volume = torch.zeros(
            (self.dim, self.dim, self.dim), dtype=torch.complex64, device=self.device
        )

        total_mux = 0
        for transducer in self.transducers:
            total_mux += transducer.t_mux

            for t in range(transducer.t_mux):
                emitter = transducer.to_complex_plane(self.ds, t)

                # plt.imshow(torch.real(emitter).detach().numpy())
                # plt.show(block=True)

                propagator = self.propagator[0]

                min_side_size = (
                    propagator.shape[-1]
                    + transducer.to_complex_plane(self.ds, 0).shape[-1]
                    - 1
                )
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
                extra = field.size()[1] - self.dim
                field = field[
                    :,
                    extra // 2 : extra // 2 + self.dim,
                    extra // 2 : extra // 2 + self.dim,
                ]

                if transducer.orientation == Orientation.Z:
                    volume += field
                elif transducer.orientation == Orientation.Z_1:
                    volume += field.rot90(2, (0, 1))
                elif transducer.orientation == Orientation.Y:
                    volume += field.rot90(-1, (0, 1))
                elif transducer.orientation == Orientation.X:
                    volume += field.rot90(-1, (0, 2))

        # print("FFTs Finished!")
        return volume.abs() / total_mux


class Orientation(Enum):
    X = 1
    Y = 2
    Z = 3
    Z_1 = 4


# WIP
class Transducer:
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
        self.phases: list[torch.Tensor]
        self.amps: list[torch.Tensor]
        self.emitter_size: float = emitter_size
        self.array_size: float = array_size
        self.pos: torch.Tensor = torch.tensor(pos, dtype=torch.float32, device=device)
        self.orientation: Orientation = orientation
        self.t_mux: int = t_mux
        self.device = device

        self.phases = [
            torch.rand(
                (emitters_num, emitters_num),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )
            for i in range(t_mux)
        ]
        self.amps = [
            torch.rand(
                (emitters_num, emitters_num),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )
            for i in range(t_mux)
        ]

    def to_complex_plane(self, ds: float, t_mux: int):
        start = time.time()
        # Input:
        # - ds: Sice of each cell. Defined by the simulation volume
        target_size = round(self.array_size / ds)
        # print(target_size)

        transducer = torch.zeros(
            (target_size, target_size), dtype=torch.complex64, device=self.device
        )

        cells_per_emitter = np.round(self.emitter_size / ds).astype(np.int32)
        gap_between = (
            target_size - cells_per_emitter * self.phases[t_mux].size()[0]
        ) // self.phases[t_mux].size()[0]

        # print(cells_per_emitter, gap_between)

        for y in range(self.phases[t_mux].size()[0]):
            for x in range(self.phases[t_mux].size()[0]):
                pos_x = int(gap_between // 2 + x * (cells_per_emitter + gap_between))
                pos_y = int(gap_between // 2 + y * (cells_per_emitter + gap_between))

                transducer[
                    pos_y : pos_y + cells_per_emitter, pos_x : pos_x + cells_per_emitter
                ] += self.amps[t_mux][y, x] * torch.exp(1j * self.phases[t_mux][y, x])

        # print(f"time took: {time.time() - start}")

        return transducer
