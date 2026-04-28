import math
import time
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .Simulation import Orientation

# WIP:
# Trying to see if something similar to ASA Convolution can be used for bruteforce method

# --- Coordinate System ---
# - In all places, the indexing of axis is [z, y, x]
# - This is kinda the standard of torch/numpy
#


# Attenuation in nP/m
# [Conversion factor dB -> Np ] * dB/m
# Termoviscoso αvt con β=0 (solo cortadura) -> 0.0118 Np/m ; 0.1 dB/m
# Termoviscoso αvt con β=1 (convención del libro) -> 0.0235 Np/m ; 0.20 dB/m
# Termoviscoso αvt con 4μ/3+μB (μB/μ≈0.6 μB/μ≈0.6 para aire, Tisza) -> 0.0227 Np/m ; 0.20 dB/m
# Atmosférico αatm, 30% HR -> 0.173 Np/m ; 1.5 dB/m
# Atmosférico αatm, 50% HR -> 0.150 Np/m ; 1.3 dB/m
# Atmosférico αatm, 70% HR -> 0.127 Np/m ; 1.1 dB/m

ATTENUATION = 1.0 / (20.0 / math.log(10)) * (1.3 + 0.2)


class Simulation_Batch:
    def __init__(
        self,
        fr,
        c,
        size=0.16,
        ds=0.16 / 64,
        device="cpu",
        optimize_amps=True,
    ):
        # Common data for all fields
        self.fr = fr
        self.c = c
        self.wavelen = c / fr
        self.k = 2 * math.pi / self.wavelen
        self.ds = ds
        self.size = size
        self.dim = int(self.size / self.ds)
        self.device = device
        self.optimize_amps = optimize_amps

        self.transducers: list[
            Transducer_Batch
        ] = []  # Array of num_emitter x num_emitter phases
        self.propagators: list[torch.Tensor] = []  # Propagator matrix
        self.slices: list[torch.Tensor] = []  # Propagator slices

        # Info for when adding transducers
        self.min_dist = np.inf
        self.max_dist = 0

    def get_params(self):
        params = []
        for tr in self.transducers:
            params += [tr.phases, tr.amps] if self.optimize_amps else [tr.phases]

        return params

    def add_transducer(self, transducer):
        same_axis = False
        for tr in self.transducers:
            same_axis = same_axis or tr.orientation == transducer.orientation

        min_dist = -transducer.pos[0]
        max_dist = min_dist + self.dim * self.ds

        self.min_dist = min(self.min_dist, min_dist)
        self.max_dist = max(self.max_dist, max_dist)

        self.transducers.append(transducer)

    def create_propagators(self):
        # Initialize transducers (creates mask)
        for transducer in self.transducers:
            transducer.init_transducer(self.ds)

        # Now create the propagator
        vertical_dim = math.ceil(float((self.max_dist - self.min_dist) / self.ds))

        # TEST: Propagator of needed size from start
        propagator = torch.zeros(
            (self.dim, self.dim * 2, self.dim * 2),
            dtype=torch.complex64,
            device=self.device,
        )

        # Create the coordinate grid
        x_coords = (
            torch.arange(-self.dim, self.dim, device=self.device).view(1, 1, -1).float()
            * self.ds
        )
        y_coords = (
            torch.arange(-self.dim, self.dim, device=self.device).view(1, -1, 1).float()
            * self.ds
        )
        z_coords = (
            torch.arange(0, vertical_dim, device=self.device).view(-1, 1, 1).float()
            * self.ds
        )  # TODO: REVISAR ESTO, EL VOLUMEN NO ESTA CENTRADO?

        # Expand dimensions for broadcasting to match the propagator shape [dim, dim, dim]
        x_coords = x_coords.expand(vertical_dim, self.dim * 2, -1)
        y_coords = y_coords.expand(vertical_dim, -1, self.dim * 2)
        z_coords = z_coords.expand(-1, self.dim * 2, self.dim * 2)

        cell_pos = torch.stack(
            [
                z_coords,
                y_coords,
                x_coords,
            ],
            dim=0,
        )

        closest_tr_pos = torch.tensor(
            [self.min_dist, 0, 0], dtype=torch.float32, device=self.device
        ).reshape(3, 1, 1, 1)

        between = cell_pos + closest_tr_pos
        dist = torch.norm(between, dim=0)

        normal_vec = torch.tensor(
            [1.0, 0.0, 0.0], device=self.device, dtype=torch.float32
        ).reshape(3, 1, 1, 1)

        dots = (between * normal_vec).sum(dim=0)

        angle = torch.arccos(dots / dist / torch.norm(normal_vec))

        # For now only one apperture value supported
        apperture = self.transducers[0].apperture
        dum = 0.5 * apperture * self.k * torch.sin(angle)

        directivity = torch.sinc(dum / torch.pi)

        propagator = (directivity / (dist + 1e-9)) * torch.exp(
            1j * (self.k * dist) - ATTENUATION * dist
        )

        self.propagators.append(propagator)

    def create_propagator_slices(self, distance):
        assert len(self.transducers) == 1, (
            "Only one transducer allowed for slice optimization"
        )

        # Initialize transducers (creates mask)
        for transducer in self.transducers:
            transducer.init_transducer(self.ds)

        # Now create the propagator
        propagator = torch.zeros(
            (1, self.dim * 2, self.dim * 2),
            dtype=torch.complex64,
            device=self.device,
        )

        # Create the coordinate grid
        x_coords = (
            torch.arange(-self.dim, self.dim, device=self.device).view(1, 1, -1).float()
            * self.ds
        )
        y_coords = (
            torch.arange(-self.dim, self.dim, device=self.device).view(1, -1, 1).float()
            * self.ds
        )

        # Expand dimensions for broadcasting to match the propagator shape [dim, dim, dim]
        x_coords = x_coords.expand(1, self.dim * 2, -1)
        y_coords = y_coords.expand(1, -1, self.dim * 2)
        z_coords = (
            torch.zeros(1, dtype=torch.float32, device=self.device)
            .view(-1, 1, 1)
            .expand(-1, self.dim * 2, self.dim * 2)
        ) + distance

        cell_pos = torch.stack(
            [
                z_coords,
                y_coords,
                x_coords,
            ],
            dim=0,
        )

        closest_tr_pos = torch.tensor(
            [self.min_dist, 0, 0], dtype=torch.float32, device=self.device
        ).reshape(3, 1, 1, 1)

        between = cell_pos + closest_tr_pos
        dist = torch.norm(between, dim=0)

        normal_vec = torch.tensor(
            [1.0, 0.0, 0.0], device=self.device, dtype=torch.float32
        ).reshape(3, 1, 1, 1)

        dots = (between * normal_vec).sum(dim=0)

        angle = torch.arccos(dots / dist / torch.norm(normal_vec))

        # For now only one apperture value supported
        apperture = self.transducers[0].apperture
        dum = 0.5 * apperture * self.k * torch.sin(angle)

        directivity = torch.sinc(dum / torch.pi)

        propagator = (directivity / (dist + 1e-9)) * torch.exp(
            1j * (self.k * dist) - ATTENUATION * dist
        )

        self.slices.append(propagator.unsqueeze(0))  # New dim for batch

    def calculate_volume(self, use_mean: bool = False):
        # For now assume there is only one for testing :P
        volume = torch.zeros(
            (self.transducers[0].t_mux, self.dim, self.dim, self.dim),
            dtype=torch.complex64,
            device=self.device,
        )

        for idx, transducer in enumerate(self.transducers):
            emitter = transducer.to_rounded_emitters(self.ds)

            (x, y, z) = emitter.size()
            emitter = emitter.reshape(x, 1, y, z)

            # TEST: Always use first propagator to save memory (there is only 1)
            propagator = self.propagators[0]

            ini_idx = round(float(-transducer.pos[0] - self.min_dist) / self.ds)
            propagator = propagator[ini_idx : ini_idx + self.dim, :, :]

            # TEST: Only pad emitter, propagator comes in needed size
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

            if transducer.orientation == Orientation.Z:
                volume += field
            elif transducer.orientation == Orientation.Z_1:
                volume += field.rot90(2, (1, 2))
            elif transducer.orientation == Orientation.Y:
                volume += field.rot90(-1, (1, 2))
            elif transducer.orientation == Orientation.X:
                volume += field.rot90(-1, (1, 3))

        if use_mean:
            return volume.abs().pow(2).sum(dim=0) / volume.shape[0]

        return (volume.abs().pow(2) / volume.shape[0]).sum(dim=0).sqrt()

    def calculate_slices(self, use_mean: bool = False):
        # For now assume there is only one for testing :P
        volume = torch.zeros(
            (self.transducers[0].t_mux, 1, self.dim, self.dim),
            dtype=torch.complex64,
            device=self.device,
        )

        transducer = self.transducers[0]
        emitter = transducer.to_rounded_emitters(self.ds)

        (B, x, y, z) = emitter.size()
        emitter = emitter.reshape(B, x, 1, y, z)

        slice = self.slices[0]

        # TEST: Only pad emitter, slice comes in needed size
        padded_emitter = F.pad(
            emitter, (self.dim // 2, self.dim // 2, self.dim // 2, self.dim // 2)
        )

        fft_emitter = torch.fft.fft2(torch.fft.ifftshift(padded_emitter))
        fft_prop = torch.fft.fft2(torch.fft.ifftshift(slice))

        convolved = fft_emitter * fft_prop

        field = torch.fft.fftshift(torch.fft.ifft2(convolved))

        # Trim the field
        extra = field.size()[-1] - self.dim
        field = field[
            :,
            :,
            :,
            extra // 2 : extra // 2 + self.dim,
            extra // 2 : extra // 2 + self.dim,
        ]

        # volume += field

        if use_mean:
            return field.abs().pow(2).sum(dim=1) / field.shape[1]

        return (field.abs().pow(2) / field.shape[1]).sum(dim=1).sqrt()

    def calculate_volume_brute(self):
        # EXTREMELY WIP: Just to confirm equality between methods
        volume = torch.zeros(
            (self.transducers[0].t_mux, self.dim, self.dim, self.dim),
            dtype=torch.complex64,
            device=self.device,
        )

        for idx, transducer in enumerate(self.transducers):
            for x in range(transducer.emitters_num):
                for y in range(transducer.emitters_num):
                    # plt.imshow(emitter[0, :, :].abs().cpu().detach().numpy())
                    # plt.show(block=True)

                    space_between = transducer.array_size / transducer.emitters_num

                    pos_z = transducer.pos[0]
                    pos_y = (
                        y - transducer.emitters_num // 2
                    ) * space_between + space_between / 2
                    pos_x = (
                        x - transducer.emitters_num // 2
                    ) * space_between + space_between / 2

                    # print(pos_x, pos_y, pos_z)

                    # TEST: Always use first propagator to save memory (there is only 1)
                    propagator = self.single_propagator(pos_z, pos_y, pos_x)

                    for tmx in range(transducer.t_mux):
                        comp_emitter = F.sigmoid(
                            transducer.amps[tmx, y, x]
                        ) * torch.exp(1j * transducer.phases[tmx, y, x])

                        field = propagator * comp_emitter

                        # Trim the field
                        extra = field.size()[-1] - self.dim
                        field = field[
                            :,
                            extra // 2 : extra // 2 + self.dim,
                            extra // 2 : extra // 2 + self.dim,
                        ]

                        if transducer.orientation == Orientation.Z:
                            volume[tmx, :, :, :] += field
                        elif transducer.orientation == Orientation.Z_1:
                            volume[tmx, :, :, :] += field.rot90(2, (0, 1))
                        elif transducer.orientation == Orientation.Y:
                            volume[tmx, :, :, :] += field.rot90(1, (0, 1))
                        elif transducer.orientation == Orientation.X:
                            volume[tmx, :, :, :] += field.rot90(1, (0, 2))

                    propagator = None

        return volume.abs().pow(2).sum(dim=0).sqrt()  # / volume.shape[0]

    def single_propagator(self, disp_z, disp_y, disp_x):
        # TEST: Propagator of needed size from start
        # Create the coordinate grid
        x_coords = (
            torch.arange(-self.dim // 2, self.dim // 2, device=self.device)
            .view(1, 1, -1)
            .float()
            * self.ds
        )
        y_coords = (
            torch.arange(-self.dim // 2, self.dim // 2, device=self.device)
            .view(1, -1, 1)
            .float()
            * self.ds
        )
        z_coords = (
            torch.arange(0, self.dim, device=self.device).view(-1, 1, 1).float()
            * self.ds
        )  # negative because plane goes down == distance increases

        # Expand dimensions for broadcasting to match the propagator shape [dim, dim, dim]
        x_coords = x_coords.expand(self.dim, self.dim, -1)
        y_coords = y_coords.expand(self.dim, -1, self.dim)
        z_coords = z_coords.expand(-1, self.dim, self.dim)

        cell_pos = torch.stack(
            [
                z_coords,
                y_coords,
                x_coords,
            ],
            dim=0,
        )

        x_coords = None
        y_coords = None
        z_coords = None

        # Ensure transducer.pos broadcasts to (3, 1, 1, 1)
        # transducer_pos_broad = transducer.pos.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        emitter_pos = torch.tensor(
            [disp_z, disp_y, disp_x], device=self.device
        ).reshape(3, 1, 1, 1)

        between = cell_pos - emitter_pos
        dist = torch.norm(between, dim=0)

        return (1.0 / (dist + 1e-9)) * torch.exp(1j * (self.k * dist))


class Transducer_Batch:
    def __init__(
        self,
        emitters_num: int,
        array_size: float,
        emitter_size: float,
        apperture: float,
        pos: list,
        orientation: Orientation = Orientation.Z,
        t_mux: int = 1,
        random_init=True,
        checkerboard=False,
        round_emitters=False,
        device="cpu",
    ):
        self.phases: torch.Tensor
        self.amps: torch.Tensor
        self.emitters_num: int = emitters_num
        self.emitter_size: float = emitter_size
        self.apperture: float = apperture
        self.array_size: float = array_size
        self.pos: torch.Tensor = torch.tensor(pos, dtype=torch.float32, device=device)
        self.orientation: Orientation = orientation
        self.t_mux: int = t_mux
        self.device = device
        self.random_init = random_init
        self.checkerboard = checkerboard
        self.round_emitters = round_emitters

        dim_x, dim_y = emitters_num, emitters_num

        if random_init:
            self.phases = torch.rand(
                (t_mux, dim_y, dim_x),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )

            self.amps = torch.rand(
                (t_mux, dim_y, dim_x),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )

        else:
            self.phases = torch.zeros(
                (t_mux, dim_y, dim_x),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )

            self.amps = torch.ones(
                (t_mux, dim_y, dim_x),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )

        with torch.no_grad():
            self.phases *= 2 * math.pi
            self.amps *= 2
            self.amps -= 1

        # Persistent info por complex plane calculation
        # Used to make calculations faster by saving information

        self.mask: torch.Tensor = None

    def reset_params(self):
        dim_x, dim_y = self.emitters_num, self.emitters_num

        if self.random_init:
            self.phases = torch.rand(
                (self.t_mux, dim_y, dim_x),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )

            self.amps = torch.rand(
                (self.t_mux, dim_y, dim_x),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )

            with torch.no_grad():
                self.phases *= 2 * math.pi
                self.amps *= 2
                self.amps -= 1

        else:
            self.phases = torch.zeros(
                (self.t_mux, dim_y, dim_x),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )

            self.amps = torch.ones(
                (self.t_mux, dim_y, dim_x),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )

            with torch.no_grad():
                self.amps *= 5

    def init_transducer(self, ds: float):
        # This creates a mask that will be used later when creating the complex plane
        # Made separately and saved to speed up execution a bit

        # Else, initialize "point" emitters

        target_size = round(self.array_size / ds)

        cells_per_emitter = target_size // self.emitters_num

        self.mask = torch.zeros(
            (target_size, target_size), dtype=torch.bool, device=self.device
        )

        self.mask[
            cells_per_emitter // 2 :: cells_per_emitter,
            cells_per_emitter // 2 :: cells_per_emitter,
        ] = True

        if self.checkerboard:
            self.mask = self.mask.roll(
                (-(cells_per_emitter // 2), -(cells_per_emitter // 2)), (0, 1)
            )

            self.mask[:: cells_per_emitter * 2, :] = self.mask[
                :: cells_per_emitter * 2, :
            ].roll(cells_per_emitter // 2, 1)

            self.mask = self.mask.roll(
                (cells_per_emitter // 2, cells_per_emitter // 4), (0, 1)
            )

        self.mask = self.mask.unsqueeze(0)  # New dim for batch

    def to_rounded_emitters(self, ds: float):
        target_size = round(self.array_size / ds)

        assert target_size % self.emitters_num == 0, (
            "Volume size must be multiple of emitter_num to allow correct spacing!"
        )

        shape = self.phases.shape

        N = target_size // shape[-1]

        complex_field = 1 * torch.exp(1j * (self.phases))  # Assume amp == 1 for now

        complex_field = (
            complex_field.repeat_interleave(N, dim=2).repeat_interleave(N, dim=3)
            * self.mask
        )

        return complex_field
