import math
from typing import List

import numpy as np
import torch

import matplotlib.pyplot as plt
from enum import Enum

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




# def emitter_field_from_phases(phases, final_size, cpe):
#     """
#     phases: (E, E) tensor with requires_grad=True
#     final_size: size of output (64)
#     cpe: cells per emitter (2)
#     """
#     device = phases.device
#     E = phases.shape[0]

#     pitch = final_size // E  # 4
#     gap = pitch - cpe        # 2

#     # Create coordinate grid
#     yy, xx = torch.meshgrid(
#         torch.arange(final_size, device=device),
#         torch.arange(final_size, device=device),
#         indexing="ij"
#     )

#     # Determine which emitter each pixel belongs to
#     emitter_y = yy // pitch
#     emitter_x = xx // pitch

#     # Determine local position inside emitter cell
#     local_y = yy % pitch
#     local_x = xx % pitch

#     # Mask for active emitter area
#     mask = (local_y < cpe) & (local_x < cpe)

#     # Clamp indices so we don't index outside
#     emitter_y = torch.clamp(emitter_y, max=E-1)
#     emitter_x = torch.clamp(emitter_x, max=E-1)

#     # Gather phase values
#     transducer = phases[emitter_y, emitter_x] * mask

#     return transducer


class SimulationASA:
    # --- Parameters ---:
    # fr: Frequency of emitters
    # c: speed of sound
    # dim: dimension in cells of each side of volume matrix
    # size: size in meters of each side of the volume matrix
    # emitter_size: diameter of each emitter

    def __init__(self, fr, c, dim=64, size=0.16, emitter_size = 0.01, distance_to_volume = 0.02):
        # Common data for all fields
        self.fr = fr
        self.c = c
        self.wavelen = c / fr
        self.k = 2 * math.pi / self.wavelen
        self.dim = dim
        self.size = size
        self.emitter_size = emitter_size
        self.distance_to_volume = distance_to_volume 

        self.transducers = [] # Array of num_emitter x num_emitter phases
        self.propagator: torch.tensor # Propagator matrix

        self.volume = torch.zeros((dim, dim), dtype=torch.complex64)

    def add_transducer(self, emitter_dim: int = 8):
        # Function:
        # - Adds a torch array of emitter_dim x emitter_dim
        # - Each value represents the phase of an emitter
        # - Has requires grads activated to allow optimization
        
        transducer = torch.zeros((emitter_dim, emitter_dim), dtype=torch.float32, requires_grad=True)
        self.transducers.append(transducer)

    def emmiter_field_from_phases(self, emitter_dim: int, phases: np.ndarray):
        assert phases.numel() == emitter_dim*emitter_dim, "Not enough phases for all emitters"
        
        cell_size = self.size / self.dim
        transducer = torch.zeros((self.dim, self.dim), dtype = torch.complex64)

        cells_per_emitter = np.round(self.emitter_size / cell_size).astype(np.int32)
        gap_between = (self.dim - cells_per_emitter*emitter_dim) // emitter_dim

        for y in range(emitter_dim):
            for x in range(emitter_dim):
                pos_x = int(gap_between//2 + x*(cells_per_emitter + gap_between))
                pos_y = int(gap_between//2 + y*(cells_per_emitter + gap_between))
                    
                transducer[pos_y:pos_y+cells_per_emitter, pos_x:pos_x+cells_per_emitter] \
                             += 1 * torch.exp(1j * phases[y, x]) 

        return transducer


    def create_propagators(self):
        self.propagator = torch.zeros((self.dim, self.dim, self.dim), dtype = torch.complex64)
        cell_size = self.size / self.dim

        for y in range(self.propagator.shape[-2]):
            for x in range(self.propagator.shape[-1]):
                for z in range(self.propagator.shape[-3]):
                    
                    pos_x = (x - self.propagator.shape[-1] // 2 ) * cell_size # + cell_size / 2 # We sum half cell because is not perfectly centered
                    pos_y = (y - self.propagator.shape[-1] // 2 ) * cell_size # + cell_size / 2
                    pos_z = self.distance_to_volume + z * cell_size

                    dist = np.sqrt(pos_x*pos_x + pos_y*pos_y + pos_z*pos_z)
                    self.propagator[z, y, x] = 1/dist * np.exp(1j * (self.k * dist))

    def calculate_volume(self):
        # For now assume there is only one for testing :P
        for transducer in self.transducers:

            emitter = self.emmiter_field_from_phases(transducer.size()[0], transducer)
            
            min_side_size = (self.propagator.shape[-1] + emitter.shape[-1] - 1)
            pad_to_size = 2**np.ceil(np.log2(min_side_size))

            pad_func = torch.nn.functional.pad
            
            prop_pad = int(np.ceil((pad_to_size - self.propagator.shape[-1]) / 2))
            padded_propagator = pad_func(self.propagator, (prop_pad, prop_pad, prop_pad, prop_pad, 0, 0))
            print(padded_propagator.size())

            emitt_pad = int(np.ceil((pad_to_size - emitter.shape[-1]) / 2))        
            padded_emitter =  pad_func(emitter, (emitt_pad, emitt_pad, emitt_pad, emitt_pad))

            fft_emitter = torch.fft.fft2(torch.fft.ifftshift(padded_emitter))
            fft_prop = torch.fft.fft2(torch.fft.ifftshift(padded_propagator))

            convolved = fft_emitter * fft_prop

            field = torch.fft.fftshift(torch.fft.ifft2(convolved))

            return field            

        


class Orientation(Enum):
    X = 1
    Y = 2
    Z = 3
    

class Transducer:
    def __init__(self, emitters_num: int, size: float, pos: list, orientation: Orientation):
        self.phases: torch.tensor
        self.size: float = size
        self.pos: list = pos
        self.orientation: Orientation = orientation

        self.phases = torch.zeros(
            (emitters_num, emitters_num),
            dtype = torch.float32,
            requires_grad = True
        )

    def to_complex_plane(self, ds: float):
        target_size = round(self.size / ds)
        
        transducer = torch.zeros((self.dim, self.dim), dtype = torch.complex64)

        cells_per_emitter = np.round(self.emitter_size / ds).astype(np.int32)
        gap_between = (self.dim - cells_per_emitter*self.phases.size()[0]) // self.phases.size()[0]

        for y in range(self.phases.size()[0]):
            for x in range(self.phases.size()[0]):
                pos_x = int(gap_between//2 + x*(cells_per_emitter + gap_between))
                pos_y = int(gap_between//2 + y*(cells_per_emitter + gap_between))
                    
                transducer[pos_y:pos_y+cells_per_emitter, pos_x:pos_x+cells_per_emitter] \
                             += 1 * torch.exp(1j * self.phases[y, x]) 

        return transducer
        
