#%%

import torch
import torchaudio

import numpy as np

import matplotlib.pyplot as plt

#%%

#file_path = '/home/cyberblaster/Audio/okarina.wav'
file_path = '/home/cyberblaster/Audio/new_okarina.wav'
#file_path = '/home/cyberblaster/Audio/okarina1.wav'
flen = 1024 * 4

audio, sr = torchaudio.load(file_path)
audio = audio[0]

a_len = len(audio) // flen


audio = audio[:a_len*flen].view(a_len, flen)

from cqt_rough import RoughCQT

cqt_fn = RoughCQT(filter_length=flen, n_filters=96, bins_per_octave=36)


cqt = []
for a in audio:
    cqt.append(cqt_fn(a.numpy()))

cqt = np.stack(cqt).T

cqt = cqt_norm = cqt / cqt.max(0)
cqt = np.nan_to_num(cqt)

plt.imshow(np.log(cqt + 1e-7))
# %%
plt.plot(cqt[:, 60:85].sum(-1))
#%%

from TheModel import Network, ConvBlock, DepthwiseConvBlock
net = torch.load('95_96_bins.ckpt')

pred = []
for c in cqt.T:
    pred.append(net(torch.from_numpy(c).unsqueeze(0).to(torch.float)))

pred = torch.stack(pred)
# %%

pred = net(torch.from_numpy(cqt.T).to(torch.float))
# %%
plt.imshow(pred.squeeze().detach().T, aspect = 'auto')

# %%
plt.imshow(torch.log(pred + 1e-7).squeeze().detach().T, aspect = 'auto')

#%%

start = 380
length = len(pred)-1 - start
end = start + length
certainty = pred.squeeze().sum(1)[start:end].detach()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (10, 6))
ax1.imshow(pred.squeeze()[start:end].T.detach(), aspect = 'auto')
ax2.plot(certainty)
ax2.plot(np.ones(length))
ax2.plot(np.ones(length) * 2)
ax2.margins(x = 0.0)
ax3.plot((certainty - 1) * (certainty - 2))
ax3.plot(np.ones(length))
ax3.margins(x = 0.0)


#%%

import torch
import numpy as np
from math import pi
import random

import matplotlib.pyplot as plt


def gen_monophonic_wave(freq, length = 1024 * 4, sr = 44100,
                        harm_max_volume = 1,
                        harm_distr_curve = 2,
                        harm_n = 5):
    
    harm_volume = np.exp(random.random() * np.log(harm_distr_curve)) / harm_distr_curve * harm_max_volume
    harms = np.array([2 ** (-i * (random.random() + 0.75)) for i in range(1, harm_n)])
    harms *= harm_volume
    freqs = np.array([freq * i for i in range(1, harm_n + 1)])
    bases = np.stack([np.linspace(0, 2*pi*f*length/sr, length) for f in freqs])
    bases += np.random.rand(harm_n).reshape(-1, 1) * 2 * pi
    wave = np.sin(bases) 
    wave[1:] *= harms.reshape(-1, 1)
    return wave.sum(0)

def gen_polyphonic_wave(freqs, length = 1024, 
                        sr = 44100, 
                        noise_range = (0, 0.2), 
                        comp_diff = 0.5,
                        debug = False):
    """ noise range: [lowest level, highest level] {0, 1} 
        comp_diff: scalar, 0 - components are equal, > 0 - components are 1 +/- comp_diff
        harm_distr_shift: {0, 1}, 0 - always the same; 1 - random normal {0, 1}
    """
    component_diff = random.random() * comp_diff
    noise_min = int(100 * noise_range[0])
    noise_max = int(100 * noise_range[1])
    noise_level = random.randint(noise_min, noise_max) / 100
    w1 = gen_monophonic_wave(freqs[0], length, sr)
    w1 = w1 * (1 + component_diff)
    w2 = gen_monophonic_wave(freqs[1], length, sr)
    w2 = w2 * (1 - component_diff)
    wave = w1 + w2 + np.random.random(length) * noise_level * 2 - 1
    wave = wave / np.abs(wave).max()
    if debug:
        return wave.reshape(1, -1), w1, w2, component_diff, noise_level
    return wave.reshape(1, -1)

from cqt_rough import RoughCQT
from notes_to_freq import SCALE

f1 = SCALE['D4']
f2 = SCALE['G4']

wave = gen_polyphonic_wave((f2, f1), length = 1024 * 4)

cqt_transform = RoughCQT(n_filters=48, bins_per_octave=24)


cqt1 = cqt_transform(wave)

plt.plot(cqt1 / cqt1.max())
plt.plot(cqt[:, 10])

#%%

