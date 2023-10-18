#%%

import numpy as np
from math import log2, pi
from .notes_to_freq import notes_to_track
from .cached_filters import CachedFilters
from .utils.create_filters import create_filters


class CQTPrecise:
    def __init__(self, length, n_filters = 9, bins_per_octave = 36, sr = 44100,
                 cached_filters = True):
        
        self.n_filters = n_filters
        self.bins_per_octave = bins_per_octave
        self.length = length
        self.sr = sr

        self.cached_filters = cached_filters

        if cached_filters:
            self.filters = CachedFilters(length, notes_to_track=notes_to_track,
                                         filters_per_octave=bins_per_octave, 
                                         n_filters=n_filters)


    def __call__(self, wave, frequencies):

        freqs = []
        vols = []

        for f in frequencies:

            if f != 0:
                freq, vol = self.detect_freq(wave, f)
                freqs.append(freq)
                vols.append(vol)
            else:
                freqs.append(0)
                vols.append(0)

        return freqs[0], freqs[1], vols[0], vols[1]
    

    def detect_freq(self, wave, freq):

        if self.cached_filters:

            filters, filter_freqs = self.filters.get_filters(freq)

        else:

            min_filter_freq = freq * 2 ** (-(self.n_filters - 1) / 2 / self.bins_per_octave)
            
            filter_freqs = np.array([min_filter_freq * 2 ** (n / self.bins_per_octave) 
                            for n in range(self.n_filters)])
            
            filters = create_filters(self.length, filter_freqs, sr = self.sr)

        cqt = np.abs(filters @ wave.T).squeeze()

        pred_freq = cqt / cqt.sum() @ filter_freqs

        vol = cqt.mean()

        return pred_freq, vol
    
#%%

# from time import time

# cqt_fn = CQTPrecise(2048, cached_filters=True)

# iters = 10000
# freqs1 = np.random.randint(320, 700, iters) + np.random.randn(iters)
# freqs2 = np.random.randint(320, 700, iters) + np.random.randn(iters)

# freq_pred_1 = []
# freq_pred_2 = []

# waves = np.array([gen_polyphonic_wave((f1, f2), length = 2048) for f1, f2 in zip(freqs1, freqs2)])

# t = time()
# for n, w in enumerate(waves):
#     f1, f2, _, _ = cqt_fn(w, (freqs1[n], freqs2[n]))
#     freq_pred_1.append(f1)
#     freq_pred_2.append(f2)

# print(time() - t)

# print((freqs1 - np.array(freq_pred_1)).mean())
# print((freqs2 - np.array(freq_pred_2)).mean())




# %%
# from polyphonic_dataset import gen_polyphonic_wave

# test_freqs = 100 * 2 ** (np.arange(1, 128) / 24)
# length = 1024 * 4

# #noise = np.random.rand(length) * 2 - 1
# wave = gen_polyphonic_wave([200,500], length = length)

# filters = create_filters(length, test_freqs)

# cqt = np.abs(filters @ wave.squeeze())

# plt.plot(cqt)

# #%%

# notes_to_track = ['D4', 'F4', 'G4', 'A4', 'C5', 'D5', 'F5']
# full_scale = get_full_scale()
# freqs = torch.tensor([full_scale[n] for n in notes_to_track])

# # test_freqs = 100 * 2 ** (np.arange(1, 100) / 30)
# # test_filter_freqs = 100 * 2 ** (np.arange(1, 128) / 24)
# # test_freqs = 350 * 2 ** (np.arange(1, 60) / 30)
# # test_filter_freqs = 350 * 2 ** (np.arange(1, 48) / 24)
# test_filter_freqs = freqs[0] * 2 ** (np.arange(150) / 75)
# test_freqs = freqs[0] * 2 ** (np.arange(48) / 24)

# length = 1024 

# cqts = []
# filters = create_filters(length, test_filter_freqs.numpy())

# for f in test_freqs:
#     wave = np.sin(np.linspace(0, 2*pi*f*length/44100, length))
#     wave += np.sin(np.linspace(0, 2*pi*f*2*length/44100, length)) 
#     wave += np.sin(np.linspace(0, 2*pi*f*3*length/44100, length)) 
#     wave /= 3
#     #wave = gen_polyphonic_wave([f, f], length, noise_range = [0,0], comp_diff = 0, harm_distr_shift=0)
#     cqt = np.abs(filters @ wave.squeeze())
#     cqts.append(cqt)

# # fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8, 12))
# # plt.imshow(cqts, aspect = 'auto')
# plt.plot(test_freqs, np.array(cqts).sum(-1))
# # ax1.plot(test_freqs, np.array(cqts).sum(-1))
# # ax2.imshow(cqts)

# cqt_fn = CustomCQT(length=2048)

#%% DAASET

# import torch
# import torch.nn as nn
# import numpy as np
# from math import pi, log
# import random

# import matplotlib.pyplot as plt

# def rand_level(range = 2):
#     r = np.exp(random.random() * (log(range ** 2) - log(1)) + log(1))
#     return r / range

# def gen_monophonic_wave(freq, length = 1024, sr = 44100,
#                         harm_distr_shift = 2):
#     harms = np.array([2 ** -i * rand_level(harm_distr_shift) for i in range(1, 5)])
#     freqs = np.array([freq * i for i in range(1, 5)])
#     bases = np.stack([np.linspace(0, 2*pi*f*length/sr, length) for f in freqs])
#     bases += np.random.rand(4).reshape(-1, 1) * 2 * pi
#     wave = np.sin(bases) * harms.reshape(-1, 1)
#     return wave.sum(0)

# def gen_polyphonic_wave(freqs, length = 1024, 
#                         sr = 44100, 
#                         noise_range = (0, 0.2), 
#                         comp_diff = 0.5,
#                         harm_distr_shift = 2):
#     component_diff = random.random() * comp_diff
#     noise_min = int(100 * noise_range[0])
#     noise_max = int(100 * noise_range[1])
#     noise_level = random.randint(noise_min, noise_max) / 100
#     w1 = gen_monophonic_wave(freqs[0], length, sr, harm_distr_shift=harm_distr_shift)
#     w1 = w1 * (1 + comp_diff)
#     w2 = gen_monophonic_wave(freqs[1], length, sr, harm_distr_shift=harm_distr_shift)
#     w2 = w2 * (1 - comp_diff)
#     wave = w1 + w2 + np.random.random(length) * noise_level
#     wave = wave / np.abs(wave).max()
#     return wave.reshape(1, -1)

# from notes_to_freq import get_scale, get_full_scale

# from torch.utils.data import Dataset, DataLoader

# bell = lambda x, l : (-((torch.arange(l) - x) * 2) ** 2).exp()

# notes_to_track = ['D4', 'F4', 'G4', 'A4', 'C5', 'D5', 'F5']
# full_scale = get_full_scale()
# freqs = torch.tensor([full_scale[n] for n in notes_to_track])

# from new_cqt import RoughCQT, cqt_slice_transform

# N_FILTERS = 48
# BINS_PER_OCTAVE = 24

# cqt_transform = RoughCQT(filter_length=2048, n_filters=N_FILTERS, bins_per_octave=BINS_PER_OCTAVE)

# class TheDataset(Dataset):
#     def __init__(self, length, amount, freqs,
#                  harm_distr_shift = 2, noise_range = (0, 0.2), comp_diff = 0.5, 
#                  pitch_shift_st = 100):
                 
#         f_index = torch.randint(0, len(freqs) - 2, (2, amount))
#         print(f_index.shape)
#         transp = torch.stack([torch.zeros(amount), torch.ones(amount) * 2]).to(torch.int)
#         print(transp.shape)
#         f_index += transp
#         freqs = torch.tensor(freqs)[f_index]
#         print(freqs.shape)
#         min_freq = cqt_transform.freqs[0]

#         self.waves = []
#         self.frequencies = []
#         self.probs = []
#         self.idx = []

#         for n, (f1, f2) in enumerate(freqs.T):

#             if pitch_shift_st != 0:
#                 p_shifts = np.random.randint(-pitch_shift_st, pitch_shift_st, 2)
#                 f1 = f1 * 2 ** (p_shifts[0] / (12 * 100))
#                 f2 = f2 * 2 ** (p_shifts[1] / (12 * 100))
            
#             wave = gen_polyphonic_wave((f1, f2), length, 
#                                        noise_range=noise_range, 
#                                        comp_diff=comp_diff,
#                                        harm_distr_shift=harm_distr_shift)
            
#             self.waves.append(torch.tensor(wave.squeeze()))
#             self.frequencies.append(torch.tensor([f1, f2]))
#             self.idx.append(f_index[:, n])
#             prob1 = bell(f_index[0][n], 7)
#             prob2 = bell(f_index[1][n], 7)
#             prob = prob1 + prob2
#             self.probs.append(prob / prob.sum())

#         self.cqt = [torch.from_numpy(cqt_transform(w.numpy())) for w in self.waves]
#         self.cqt = torch.stack(self.cqt)
#         self.frequencies = torch.stack(self.frequencies)
#         self.waves = torch.stack(self.waves)
#         self.probs = torch.stack(self.probs)
#         self.idx = torch.stack(self.idx)

#     def __getitem__(self, index):
#         return (self.waves[index], 
#                 self.frequencies[index], 
#                 self.probs[index], 
#                 self.cqt[index],
#                 self.idx[index])
    
#     def __len__(self):
#         return len(self.frequencies)

# dataset = TheDataset(2048, 20, freqs)#, noise_range=[0, 0], comp_diff=0, harm_distr_shift=1)

# %%

# from dataset import TheDataset
# import random
# from notes_to_freq import SCALE, notes_to_track

# dataset = TheDataset(1024 * 2, 100, freqs = None)

# freqs = [SCALE[n] for n in notes_to_track]


# r = random.randint(0, len(dataset)-1)

# (f1, f2), prob, cqt, i = dataset[r]

# freqs_init_pred = freqs[i]

# analyse = cqt_fn(wave.numpy(), freqs_init_pred.numpy(), 0.5, test=True)

# freq1, freq2, level, freqs1, freqs2, cqt1, cqt2, spread = analyse

# print('true frequencies:', *[freq[0].round().item(), freq[1].round().item()])
# print('prediction      :', round(freq1, 2), round(freq2, 2))
# print('hypothesis      :', *[freqs_init_pred[0].round().item(), freqs_init_pred[1].round().item()])
# print(freqs1.round(2))
# print(freqs2.round(2))
# plt.plot(cqt1)
# plt.plot(cqt2)
# # %%

# new_hypothesis = [np.copy(freq1), np.copy(freq2)]
# analyse = cqt_fn(wave.numpy(), [freq1, freq2], 0.2, test=True)
# freq1, freq2, level, freqs1, freqs2, cqt1, cqt2, spread = analyse

# print('true frequencies:', *[freq[0].round().item(), freq[1].round().item()])
# print('prediction      :', *[round(freq1, 2), round(freq2, 2)])
# print('hypothesis      :', *[new_hypothesis[0].round().item(), new_hypothesis[1].round().item()])
# print(freqs1.round(2))
# print(freqs2.round(2))
# plt.plot(cqt1)
# plt.plot(cqt2)
# print()

# #%%
# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.imshow(cqts)
# ax2.plot(np.array(cqts).sum(-1))
# # %%

# plt.plot(filters[5].real)
# plt.plot(filters[80].real)
# print(filters[5].real.sum(), filters[80].real.sum())
# # %%

# %%

# class CustomCQT:
#     def __init__(self, 
#                  length = 1024,
#                  sr = 22050,
#                  n_filters = 7):
        
#         self.n_filters = n_filters
#         self.length = length
#         self.sr = sr
#         self.filters_1 = None
#         self.filters_2 = None
#         self.freqs_1 = None
#         self.freqs_2 = None

#     def decode_cqt(self, cqt, freqs):
#         return (cqt/cqt.sum() * freqs).sum()
        
#     def __call__(self, wave, spread_range = 2, test = False, freqs = None):

#         if freqs is not None:
#             f1, f2 = freqs
#             spread = np.arange(-self.n_filters//2 + 1, self.n_filters//2 + 1) * spread_range
#             self.freqs_1 = f1 * 2 ** (spread / 24)
#             self.freqs_2 = f2 * 2 ** (spread / 24)
            
            
#             self.filters_1 = create_filters(self.length, self.freqs_1, sr = self.sr)
#             self.filters_2 = create_filters(self.length, self.freqs_2, sr = self.sr)

#             ### LOUDNESS COEFFICIENT : CAN BE MADE BETTER
#             # test_wave1 = np.sin(np.linspace(0, 2 * pi * freqs[0] * self.length / self.sr, self.length))
#             # test_wave2 = np.sin(np.linspace(0, 2 * pi * freqs[1] * self.length / self.sr, self.length))
#             # prop1 = np.abs(self.filters_1 @ test_wave1).sum()
#             # prop2 = np.abs(self.filters_2 @ test_wave2).sum()
        
#         cqt1 = np.abs(self.filters_1 @ wave) #/ prop1
#         cqt2 = np.abs(self.filters_2 @ wave)# / prop2

#         level1 = cqt1.mean()
#         level2 = cqt2.mean()

#         freq1 = self.decode_cqt(cqt1, self.freqs_1)
#         freq2 = self.decode_cqt(cqt2, self.freqs_2)

#         if not test:
#             return freq1, freq2, level1, level2
#         else:
#             return freq1, freq2, level1, level2, f1, f2, cqt1, cqt2, spread