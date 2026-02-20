import numpy as np
import math
import os
import matplotlib.pyplot as plt
import scipy as sp
import time
from PIL import Image

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
        
def main():
    print("Entering main")

    # Simulation constants
    fr = 40e3
    c = 343.0
    amp = 1.0
    phase = 0.0

    wavelen = c / fr
    k = 2 * math.pi / wavelen

    print("Wavelength: ", wavelen)
    print("Wavenumber: ", k)

    # Simulation planes
    #  - Emitter amplitude and phase plane
    #  - Propagator plane with distance embedded
    emitter_plane = np.zeros([128, 128], dtype=np.complex64)
    propagator = np.zeros([2, 128, 128], dtype=np.complex64)


    # Calculations
    prop_side = 0.16 # 10x10 CM propagator field
    prop_h = 0.05 # How far is the propagator slice
    cell_size = prop_side / propagator.shape[-1]

    emitter_diameter = 0.01
    cells_per_emitter = np.floor(emitter_diameter / cell_size)
    print("Emitter dim: ", cells_per_emitter)
    
    for y in range(4, emitter_plane.shape[0], 8):
        for x in range(4, emitter_plane.shape[1], 8):
            emitter_plane[y:y+1, x:x+1] = amp * np.exp(1j * phase)

    plt.imshow(np.abs(emitter_plane))
    plt.show(block=True)

    
    for y in range(propagator.shape[-2]):
        for x in range(propagator.shape[-1]):
            
            pos_x = (x - propagator.shape[-1] // 2 ) * cell_size + cell_size / 2 # We sum half cell because is not perfectly centered
            pos_y = (y - propagator.shape[-1] // 2 ) * cell_size + cell_size / 2
            pos_z = prop_h

            dist = np.sqrt(pos_x*pos_x + pos_y*pos_y + pos_z*pos_z)
            propagator[0, y, x] = 1/dist * np.exp(1j * (k * dist))
            
            dist = np.sqrt(pos_x*pos_x + pos_y*pos_y + (pos_z+cell_size)**2)
            propagator[1, y, x] = 1/dist * np.exp(1j * (k * dist))


    plt.imshow(np.clip(np.real(propagator[0, :, :]), 0, 1))
    plt.show(block=True)    # Final slice calculation

    min_side_size = (propagator.shape[-1] + emitter_plane.shape[-1] - 1)
    pad_to_size = 2**np.ceil(np.log2(min_side_size))

    print(min_side_size)
    print(pad_to_size)

    padded_emitter = np.pad(
                            emitter_plane,
                            int(np.ceil((pad_to_size - emitter_plane.shape[-1]) / 2)),
                            pad_with,
                            padder=0
                        )
    
    # padded_propagator = np.pad(
    #                         propagator,
    #                         int(np.ceil((pad_to_size - propagator.shape[0]) / 2)),
    #                         pad_with,
    #                         padder=0
    #                     )


    padding = int(np.ceil((pad_to_size - propagator.shape[-1]) / 2))
    padded_propagator = np.pad(
                            propagator,
                            ((0, 0), (padding, padding), (padding, padding)),
                            mode="constant",
                            constant_values=0+0j
                        )

    print(padded_propagator.shape)

    # plt.imshow(np.abs(padded_emitter))
    # plt.show(block=True)
    # plt.imshow(np.abs(padded_propagator))
    # plt.show(block=True)

    start = time.time() 
    fft_emitter = np.fft.fft2(np.fft.ifftshift(padded_emitter))
    fft_prop = np.fft.fft2(np.fft.ifftshift(padded_propagator[:, :]) )
   
    convolved = fft_emitter * fft_prop

    field = np.fft.fftshift(np.fft.ifft2(convolved))

    duration = time.time() - start
    print("FFT Convolution Duration: ", duration)
    
    plt.imshow(np.real(fft_emitter))
    plt.show(block=True)
    plt.imshow(np.clip(np.real(padded_propagator[0, :, :]), 0, 1))
    plt.show(block=True)

    # convolved = sp.signal.convolve2d(padded_emitter, propagator, boundary="wrap")

    # plt.imshow(np.abs(convolved)/400)
    plt.imshow(np.clip(np.abs(field[0, :, :]) / 400, 0, 5))
    plt.show(block=True)



    os.exit(-1)


    # Get initial from amplitudes

    image = Image.open("./smiley.png")
    image = image.convert('L')

    target_field = (np.array(image).astype(np.float32) / 255).astype(np.complex64)
    target_field = np.pad(
                            target_field,
                            int(np.ceil((pad_to_size - target_field.shape[0]) / 2)),
                            pad_with,
                            padder=0
                        )

    a = np.fft.fft2(target_field)
    noise_estimate = 1e-10 
    wiener_filter = np.conj(fft_prop) / (np.abs(fft_prop)**2 + noise_estimate)
    b = a * wiener_filter
    reversed = np.fft.ifft2(b)
    
    plt.imshow(np.real(reversed))
    plt.show(block=True)

        

if __name__ == "__main__":
    main()
