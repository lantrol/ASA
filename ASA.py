import math
from typing import List
import time

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

# Coordinate System:
# - X: Horizontal axis
# - Y: Depth axis
# - Z: Vertical axis
#
# WARNING: Volume indexes are later defined as [Z, Y, X]
# - This is since the FFT convolution requires the paralel plane
#   to be in the last dimensions



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

    def __init__(self, fr, c, size=0.16, pos: list[float] = [0, 0, 0.09], ds= 0.16 / 64):
        # Common data for all fields
        self.fr = fr
        self.c = c
        self.wavelen = c / fr
        self.k = 2 * math.pi / self.wavelen
        self.pos = torch.tensor(pos, dtype = torch.float32)
        self.ds = ds
        self.size = size
        self.dim = int(self.size/self.ds)

        self.transducers: list[Transducer] = [] # Array of num_emitter x num_emitter phases
        self.propagator: list[torch.tensor] = [] # Propagator matrix
        
        self.volume = torch.zeros((self.dim, self.dim), dtype=torch.complex64)

    def add_transducer(self, transducer):    
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

        start = time.time()
        print("Starting the propagator creation...")
        for transducer in self.transducers:
            propagator = torch.zeros((self.dim, self.dim, self.dim), dtype = torch.complex64)
            
            for z in range(propagator.shape[2]):
                for y in range(propagator.shape[1]):
                    for x in range(propagator.shape[0]):
                        cell_pos = torch.tensor(
                            [
                                (x - self.dim // 2) * self.ds + self.pos[0],
                                (y - self.dim // 2) * self.ds + self.pos[1],
                                (z - self.dim // 2) * self.ds + self.pos[2],
                            ]
                        )

                        between = cell_pos - transducer.pos
                        dist = between.norm()
                        propagator[z, y, x] = dist
                        # propagator[z, y, x] = 1/dist * np.exp(1j * (self.k * dist))

            propagator = 1/propagator * torch.exp(1j * (self.k * propagator))

        end = time.time()
        print(f"Finished propagators creation in {end - start} seconds")

        self.propagator.append(propagator)
        
    
    def calculate_volume(self):
        # For now assume there is only one for testing :P
        for transducer in self.transducers:

            # emitter = self.emmiter_field_from_phases(transducer.size()[0], transducer)

            emitter = transducer.to_complex_plane(self.ds)
            propagator = self.propagator[0]

            plt.imshow(torch.abs(emitter).detach().numpy())
            plt.show(block=True)
            
            min_side_size = (propagator.shape[-1] + emitter.shape[-1] - 1)
            pad_to_size = 2**np.ceil(np.log2(min_side_size))

            pad_func = torch.nn.functional.pad
            
            prop_pad = int(np.ceil((pad_to_size - propagator.shape[-1]) / 2))
            padded_propagator = pad_func(propagator, (prop_pad, prop_pad, prop_pad, prop_pad, 0, 0))
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
    
# WIP
class Transducer:
    def __init__(self, emitters_num: int, array_size:float, emitter_size: float, pos: list, orientation: Orientation = Orientation.Z):
        self.phases: torch.tensor
        self.amps: torch.tensor
        self.emitter_size: float = emitter_size
        self.array_size: float = array_size
        self.pos: list = torch.tensor(pos, dtype = torch.float32)
        self.orientation: Orientation = orientation

        self.phases = torch.zeros(
            (emitters_num, emitters_num),
            dtype = torch.float32,
            requires_grad = True
        )
        self.amps = torch.ones(
            (emitters_num, emitters_num),
            dtype = torch.float32,
            requires_grad = True
        )

    def to_complex_plane(self, ds: float):
        # Input:
        # - ds: Sice of each cell. Defined by the simulation volume
        target_size = round(self.array_size / ds)
        print(target_size)
        
        transducer = torch.zeros((target_size, target_size), dtype = torch.complex64)

        cells_per_emitter = np.round(self.emitter_size / ds).astype(np.int32)
        gap_between = (target_size - cells_per_emitter*self.phases.size()[0]) // self.phases.size()[0]

        print(cells_per_emitter, gap_between)
        
        for y in range(self.phases.size()[0]):
            for x in range(self.phases.size()[0]):
                pos_x = int(gap_between//2 + x*(cells_per_emitter + gap_between))
                pos_y = int(gap_between//2 + y*(cells_per_emitter + gap_between))
                    
                transducer[pos_y:pos_y+cells_per_emitter, pos_x:pos_x+cells_per_emitter] \
                             += self.amps[y, x] * torch.exp(1j * self.phases[y, x]) 

        return transducer
        
