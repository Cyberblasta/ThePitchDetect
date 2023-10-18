import numpy as np
from math import pi

def create_filters(length, freqs, sr = 44100):

    bases = np.linspace(0, 2 * pi * length / sr, length) * 1j
    bases = np.tile(bases, (len(freqs), 1))
    
    filters = np.exp(bases * freqs.reshape(-1, 1))
    filters *= np.hanning(length)

    return filters