
#%%
import numpy as np
#import matplotlib.pyplot as plt
from math import pi, log2
import random

from .notes_to_freq import get_scale, get_full_scale

#%%

def create_filters(length, freqs, sr = 44100, dtype = np.complex64):

    bases = np.linspace(0, 2 * pi * length / sr, length, dtype=dtype) * 1j
    bases = np.tile(bases, (len(freqs), 1))

    filters = np.exp(bases * freqs.reshape(-1, 1), dtype=dtype)
    filters *= np.hanning(length)

    return filters

def create_cqt_filters(length, filter_ratios, freqs, hi_frequencies, low_frequencices, 
                       sr = 44100, dtype = np.complex64):

    low_freqs = freqs[freqs < low_frequencices]
    mid_freqs = freqs[(freqs > low_frequencices) & (freqs < hi_frequencies)]
    hi_freqs = freqs[freqs > hi_frequencies]

    lengths = (length // r for r in filter_ratios)
    #filters = map(create_filters, lengths, (low_freqs, mid_freqs, hi_freqs))
    filters = [create_filters(length, fs, sr=sr, dtype=dtype) 
               for length, fs in zip(lengths, [low_freqs, mid_freqs, hi_freqs])]
    
    return filters

# filters = create_cqt_filters(1024, [1,1,2], np.array([100,200,300,400,500,600,700])
#                              , hi_frequencies=600,
#                              low_frequencices=400, dtype = np.complex64)


#%%

#%%

class RoughCQT:
    def __init__(self,
                 sr = 44100,
                 filter_length = 4096,
                 filter_ratios = [1,2,2],
                 hi_frequencies = 1000,
                 low_frequencies =  300,
                 bins_per_octave = 12,
                 lowest_note = 'C4',
                 highest_note = None,
                 n_filters = 36,
                 freqs = None,
                 filter_dtype = np.complex64,
                 output_dtype = np.float32):
        
        self.filter_length = filter_length
        self.filter_ratios = filter_ratios
        self.hi_frequencies = hi_frequencies
        self.low_frequencies = low_frequencies
        self.n_filters = n_filters
        self.sr = sr
        self.output_dtype = output_dtype

        if freqs is not None:
            self.freqs = freqs
        
        else:
            scale = get_scale(n_filters, lowest_note=lowest_note)
            lowest_freq = scale[lowest_note]
            self.freqs = lowest_freq * 2 ** (np.arange(n_filters) / bins_per_octave)
        
        self.filter_ratios = filter_ratios

        self.filters = create_cqt_filters(length = filter_length,
                                          filter_ratios=filter_ratios,
                                          freqs = self.freqs,
                                          hi_frequencies=hi_frequencies,
                                          low_frequencices=low_frequencies,
                                          sr = sr, dtype = filter_dtype)
        
    def wave_reshape(self, wave): ##### REMOVE DOUBLE RESHAPE WITH cqt_transform FUNCTION
        return (wave.reshape(r, -1) for r in self.filter_ratios)

    def apply_filters(self, filters, wave):
        return np.abs(filters @ wave.T).mean(-1).astype(self.output_dtype) / self.n_filters

    def __call__(self, waves):
        
        waves = self.wave_reshape(waves)

        cqt = list(map(self.apply_filters, 
                       self.filters,
                       waves))

        return np.concatenate(cqt, axis = 0)
    
    def transform_given_freqs(self, waves, freqs):

        self.filters = create_cqt_filters(length = self.fillter_length,
                                    filter_ratios=self.filter_ratios,
                                    freqs = freqs,
                                    hi_frequencies=self.hi_frequencies,
                                    low_frequencices=self.low_frequencies,
                                    sr = self.sr)
        
        waves = self.wave_reshape(waves)

        cqt = list(map(self.apply_filters, 
                       self.filters,
                       waves))

        return np.concatenate(cqt, axis = 0)
    
def cqt_slice_transform(wave,
                  cqt_fn,
                  overlap = True):

    assert wave.shape[-1] == 1024 and wave.shape[0] >= 4, 'wave is too small'

    if wave.shape[0] != 4: ##### REMOVE DOUBLE RESHAPE WITH RoughCQT.wave_reshape
        if overlap:
            seg_number = len(wave) - 3
            new_wave = np.empty((seg_number, 4, 1024))
            for s_n in range(seg_number):
                new_wave[s_n] = wave[s_n : s_n + 4]
        else:
            raise NotImplementedError('To be implenmented...')
        
        cqt = np.empty((seg_number, len(cqt_fn.freqs)))
        for n, w in enumerate(new_wave):
            cqt[n] = cqt_fn(w)

    else:
        cqt = cqt_fn(wave)

    return cqt.T

    #filters = map(create_filters, lengths, (low_freqs, mid_freqs, hi_freqs))

#%%

""" DTYPE SPEED TEST """

# from time import time

# dtype = np.float32

# cqt_fn = RoughCQT(output_dtype=dtype)

# waves = np.random.rand(1000, 1, 4096)

# t = time()

# for w in waves:
#     cqt = cqt_fn(w)

# print(time() - t)

# %%

# sr  = 44100
# length = 1024 * 4
# freqs = np.array([100,200,300,400,500,600,700])
# dtype = np.complex64

# bases = np.linspace(0, 2 * pi * length / sr, length, dtype=dtype) * 1j
# bases = np.tile(bases, (len(freqs), 1))
# print(bases.dtype, freqs.dtype)
# filters = np.exp(bases * freqs.reshape(-1, 1), dtype = np.complex64)

# print(filters.dtype)
# %%
