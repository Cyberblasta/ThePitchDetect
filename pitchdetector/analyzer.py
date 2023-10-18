#%%

import numpy as np
from math import pi

from .notes_to_freq import SCALE, add_margin, cents_to_freq, freq_to_cents

from .cqt_rough import RoughCQT
from .cqt_precise import CQTPrecise
from .message import Message1
from .config import config
from .network import net

def get_level(audio, mode = 'mean'):
    if mode == 'mean':
        level = np.abs(audio).mean() * pi / 2
    if mode == 'max':
        level = audio.max()
    return level

def get_harm_noise_ratio(cqt):
    quantile = np.quantile(cqt, 0.5)
    max = cqt.max()
    ratio = np.log(max / quantile + 1e-7)
    return ratio

def check_pred_volumes(vol1, vol2, threshold):
    check = lambda vol : 0 < vol < threshold
    if check(vol1) or check(vol2):
        return True
    else:
        return False

def note_decode(note_probs, threshold):
    notes = np.arange(len(note_probs))
    notes[note_probs < threshold] = -1
    notes = notes[note_probs.argsort()]
    return notes[-2:]

def note_to_freqs(notes, freqs):
    ff = np.array([freqs[n] if n >= 0 else 0 for n in notes])
    return ff

def freq_to_closest_note(freq, border_freqs):
    if freq == 0:
        return -1
    for n, bf in enumerate(border_freqs[1:-1]):
        if freq < bf:
            return n
    return n

def freqs_to_closest_notes(freqs, border_freqs):
    notes = np.array([freq_to_closest_note(f, border_freqs) for f in freqs])
    return notes

def notes_to_probs(notes):
    probs = np.zeros(8)
    probs[notes[0]] = 1 if (notes[0] != notes[1]) and (notes[0] != -1) else 0
    probs[notes[1]] = 1 if (notes[1] != -1) else 0
    return probs


freqs = np.array([SCALE[n] for n in config.notes_to_track])

bell = lambda x, mu, sigma : -np.exp(-sigma*(x + mu)**2) + 1

def calc_uncertainty(probs, method = 'max'):
    if method == 'square':
        return ((probs - 1) * (probs - 2))
    if method == 'max':
        return min(1 - np.clip(probs, 0, 0.5), np.abs(np.clip(probs, 1.5, None) - 2))
    if method == 'exp':
        uncertainty = min(bell(probs, -1, 10), bell(probs, -2, 10))
        return uncertainty if probs != 0 else 0


#%%



class Analyzer:

    def __init__(self, sr = 44100, length = 1024 * 2, 
                 rough_bins_per_octave = 36, 
                 rough_n_filters = 96,
                 precise_bins_per_octave = 36,
                 precise_n_filters = 10,
                 debug = False,
                 level_threshold = 0.04,
                 harm_noise_threshold = 3,
                 note_decode_threshold = 0.2,
                 precise_threshold = 1,
                 uncertainty_threshold = 0.3,
                 cached_precise_filters = True):
        
        self.debug = debug

        self.level_threshold = level_threshold
        self.harm_noise_threshold = harm_noise_threshold
        self.note_decode_threshold = note_decode_threshold
        self.precise_threshold = precise_threshold
        self.uncertainty_threshold = uncertainty_threshold

        self.cqt_rough = RoughCQT(sr = sr, 
                             filter_length=length, 
                             filter_ratios=[1,1,1], 
                             bins_per_octave=rough_bins_per_octave,
                             n_filters=rough_n_filters)

        self.cqt_precise = CQTPrecise(length = length, sr = sr, 
                                     n_filters=precise_n_filters,
                                     bins_per_octave=precise_bins_per_octave,
                                     cached_filters=cached_precise_filters)


        self.note_pred_net = net
        #self.note_pred_net.eval()
        self.freqs = [None, None]
        self.notes = None
        self.note_probs = None
        self.uncertainty = None

        _, cents = add_margin(config.notes_to_track, 2, upper_margin=True, lower_margin=True)
        mid_cents = cents[:-1] + np.diff(cents) // 2
        self.mid_freqs = cents_to_freq(mid_cents)

        self.border_freqs = cents_to_freq(cents)
        self.ideal_freqs = np.array([SCALE[n] for n in config.notes_to_track])

        self.refresh = True

    def analyse(self, audio):

        msg = Message1()

        level = get_level(audio, mode = 'mean')
        if level < self.level_threshold:
            self.refresh = True
            return msg.add({'refresh': True, 'level': level})
        
        refresh = self.refresh
        harm_noise_ratio = 0
        

        if self.refresh:
            r_cqt = self.cqt_rough(audio)        
            
            harm_noise_ratio = get_harm_noise_ratio(r_cqt)
            if harm_noise_ratio < self.harm_noise_threshold:
                return msg.add({'level': level, 'noise': harm_noise_ratio})

            r_cqt = r_cqt.reshape(1, 96)
            r_cqt = r_cqt / r_cqt.max()

            self.note_probs = self.note_pred_net(r_cqt).squeeze()
            self.uncertainty = calc_uncertainty(self.note_probs.sum(), method = 'exp')

            self.notes = note_decode(self.note_probs, self.note_decode_threshold)
            self.notes = np.flip(np.sort(self.notes))

            self.freqs = note_to_freqs(self.notes, self.mid_freqs)

            if self.uncertainty > self.uncertainty_threshold:
                self.refresh = True
            else:
                self.refresh = False

        f1, f2, vol1, vol2 = self.cqt_precise(audio, self.freqs)

        new_notes = freqs_to_closest_notes((f1, f2), self.border_freqs)
    
        if new_notes[0] == new_notes[1]:
            new_notes[1] = -1
            f2 = 0
            self.notes = new_notes


        if self.notes[0] not in new_notes or self.notes[1] not in new_notes:
            self.notes = new_notes
            #self.refresh = True

        self.freqs = np.array([f1, f2])
        
        self.goal_freqs = np.array([self.ideal_freqs[n] if n != -1 else 0 for n in self.notes])
        self.goal_cents = np.array([freq_to_cents(f) if f != 0 else 0 for f in self.goal_freqs])
        self.cents = np.array([freq_to_cents(f) if f != 0 else 0 for f in [f1, f2]])
        errors = np.array([p - t for p, t in zip(self.goal_cents, self.cents)])
        #errors = np.array([freq_to_cents(e) for e in errors])

        if not self.refresh:
            self.refresh = check_pred_volumes(vol1, vol2, self.precise_threshold)


        return msg.add({'freqs': self.freqs, 'notes': self.notes, 'errors': errors,
                        'vols': [vol1, vol2], 'probs': self.note_probs, 
                        'uncertainty': self.uncertainty, 'refresh': self.refresh,
                        'level': level, 'noise': harm_noise_ratio})


#%% LOAD REAL WAVE

""" TESTING WITH REAL WAVE"""

# import torchaudio
# #file_path = '/home/cyberblaster/Audio/okarina.wav'
# #file_path = '/home/cyberblaster/Audio/new_okarina.wav'
# file_path = '/home/cyberblaster/Audio/re-recorded_ocarina.mp3'
# #file_path = '/home/cyberblaster/Audio/okarina1.wav'
# flen = 1024 * 2

# audio, sr = torchaudio.load(file_path)
# audio = audio[0]

# a_len = len(audio) // flen


# audio = audio[:a_len*flen].view(a_len, flen)


# #%%

# ## NO DEBUG TEST

# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from message import Messages

# analyser = Analyzer(level_threshold=0.04, harm_noise_threshold=1)

# msgs = Messages()

# with tqdm(audio) as aud:
#     for a in aud:
#         result = analyser.analyse(a.numpy())
#         if result is not None:
#             msgs.add(result)

# msgs.numpyfy()
            
# #%%

# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize = (10, 8))
# ax1.plot(msgs.freqs[:, 0])
# ax1.plot(msgs.freqs[:, 1])
# ax1.margins(x=0)
# ax2.plot(msgs.notes[:, 0])
# ax2.plot(msgs.notes[:, 1])
# ax2.margins(x=0)
# ax3.plot(msgs.errors[:, 0])
# ax3.plot(msgs.errors[:, 1])
# ax3.margins(x=0)
# ax3.set_ylim(-300, 300)
# ax4.plot(msgs.uncertainty)
# ax4.margins(x=0)
# ax5.imshow(msgs.probs.T, aspect = 'auto')

# #%%

# noisy_cqt = np.stack([analyser.cqt_rough(a.numpy()) for a in audio]).T
# noisy_cqt = noisy_cqt / noisy_cqt.max(0)
# preds = np.stack([analyser.note_pred_net(c.reshape(1, -1)) for c in noisy_cqt.T]).squeeze()

# #%%
# noisy_cqt = np.stack([analyser.cqt_rough(a.numpy()) for a in audio]).T
# noise = noisy_cqt[:, noisy_cqt.sum(0) < np.quantile(noisy_cqt.sum(0), 0.1)]
# noisy_cqt = noisy_cqt / noisy_cqt.max(0)

# cqt_no_noise = noisy_cqt - noise.mean(1).reshape(-1, 1)
# cqt_no_noise -= cqt_no_noise.min()
# cqt_no_noise /= cqt_no_noise.max(0)

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.imshow(noisy_cqt, aspect = 'auto')
# ax2.imshow(cqt_no_noise, aspect = 'auto')
# #%%

# preds = np.stack([analyser.note_pred_net(c.reshape(1, -1)) for c in noisy_cqt.T]).squeeze()
# no_noise_preds = np.stack([analyser.note_pred_net(c.reshape(1, -1)) for c in cqt_no_noise.T]).squeeze()

# #%%

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.imshow(preds.T, aspect = 'auto')
# ax2.imshow(no_noise_preds.T, aspect = 'auto')
# #%%

# plt.imshow((preds - no_noise_preds).T, aspect = 'auto')

# #%%

# notes = np.stack([note_decode(p, 0.2) for p in preds])
# no_noise_notes = np.stack([note_decode(p, 0.2) for p in no_noise_preds])

# def get_note_preds(notes):
#     note_preds = []
#     for frame in notes:
#         note_pred = np.zeros(8)
#         note_pred[frame[0]] = 1 if frame[0] != -1 else 0
#         note_pred[frame[1]] = 1 if frame[1] != -1 else 0
#         note_preds.append(note_pred)
#     return np.stack(note_preds)

# note_preds = get_note_preds(notes)
# no_noise_note_preds = get_note_preds(no_noise_notes)


# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.imshow(note_preds.T, aspect = 'auto')
# ax2.imshow(no_noise_note_preds.T, aspect = 'auto')










#%%

# import cProfile, pstats
# from wave_generator import gen_polyphonic_wave



# def profile(audio):
#     msgs = []
#     refresh = True
#     for a in audio:
#         result = analyser.analyse(a.numpy())
#         msgs.append(result)

# profiler = cProfile.Profile()
# profiler.enable()
# profile(audio)
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('cumtime')
# stats.print_stats()


# analyser = Analyzer(debug = True, level_threshold=0.04, harm_noise_threshold=1)

# detected_notes = []
# freqs1 = []
# freqs2 = []
# refreshes = []
# refresh = True
# all_volumes = []
# all_note_probs = []
# unsertainties = []

# from tqdm import tqdm

# with tqdm(audio) as aud:
#     for a in aud:
#         result = analyser.analyse(a.numpy())
#         if result is not None:
#             f1, f2, refresh, notes, volumes, note_probs, unsertainty = result
#             freqs1.append(f1)
#             freqs2.append(f2)
#             #all_note_probs.append(note_probs.detach().numpy().squeeze())
#             all_note_probs.append(note_probs.squeeze())
#             all_volumes.append(volumes)
#             unsertainties.append(unsertainty)
#         else:
#             detected_notes.append(np.zeros(8))
#             freqs1.append(0)
#             freqs2.append(0)
#             refreshes.append(False)
#             all_note_probs.append(np.zeros(8))
#             all_volumes.append(np.array([0, 0]))
#             unsertainties.append(0)
#             continue
#         refreshes.append(refresh)
#         n1_probs = np.zeros(8)
#         n1_probs[notes[0]] = 1 if notes[0] != -1 else 0
#         n2_probs = np.zeros(8)
#         n2_probs[notes[1]] = 1 if notes[1] != -1 else 0
#         note_probs = n1_probs + n2_probs
#         note_probs = note_probs/note_probs.max()
#         detected_notes.append(note_probs)


#%%

# all_note_probs = np.stack(all_note_probs)

# start = 0
# unc = [calc_uncertainty(z, method='exp') for z in all_note_probs.sum(1)]
# end = len(unc)
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = (10, 7))
# ax1.imshow(all_note_probs.squeeze()[start:end].T, aspect = 'auto')
# ax2.plot(all_note_probs.sum(1)[start:end])
# ax2.margins(x=0)
# ax2.grid()
# ax3.plot(unc[start:end]) 
# ax3.margins(x=0)
# ax3.plot(np.ones(len(unc))*analyser.uncertainty_threshold)
# ax3.grid()
# ax4.plot(refreshes[start:end])
# ax4.margins(x=0)
# ax4.grid()


#%%
# fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize = (12, 10))
# cqt = np.stack([analyser.cqt_rough(a.numpy()) for a in audio])

# start = 0
# length = len(audio)
# end = start + length

# detected_notes = np.stack(detected_notes)
# volumes = np.stack(all_volumes)

# ax1.imshow(detected_notes[start:end].T, aspect = 'auto', origin = 'lower')
# ax3.imshow(cqt[start:end].T, aspect = 'auto', origin = 'lower')
# ax2.imshow(all_note_probs[start:end].T, aspect = 'auto', origin = 'lower')
# ax4.plot(freqs1[start:end])
# ax4.plot(freqs2[start:end])
# ax4.margins(x=0.0)
# ax6.plot(refreshes[start:end])
# ax6.margins(x=0.0)
# ax5.plot(volumes[:, 0][start:end])
# ax5.plot(volumes[:, 1][start:end])
# ax5.margins(x=0.0)
# ax7.plot(unsertainties[start:end])
# ax7.margins(x = 0.0)



#%%
# start = 0
# length = len(all_note_probs)
# end = start + length
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (10, 6))
# ax1.imshow(all_note_probs[start:end].T, aspect = 'auto')
# ax2.plot(all_note_probs[start:end].sum(1))
# ax2.margins(x=0.0)
# ax3.imshow(cqt[start:end].T, aspect = 'auto')

#%%
# vol = np.abs(audio.numpy()).mean(-1) * pi / 2
# plt.plot(vol[:400])

#%%
# from time import time
# from tqdm import tqdm

# audio = audio[:300]


# def profile():

#     prds = []
#     cqts = []
#     with torch.no_grad():
#         for a in audio:
#             if np.abs(a).mean() * pi / 2 > 0.04:
#                 cqt = analyser.cqt_rough(a)
#                 cqt = cqt/cqt.max()
#                 cqt = torch.from_numpy(cqt).unsqueeze(0).to(torch.float).contiguous()
#                 #print(cqt.dtype, cqt.shape)
#                 p = network(cqt)
#                 prds.append(p.detach().numpy().squeeze())
#                 cqts.append(cqt)
#             # else:
#             #     prds.append(np.zeros(8))
#             #     cqts.append(np.zeros(96))

#     prds = np.stack(prds)
#     cqts = np.stack(cqts)
#     return prds

# t = time()
# results = profile()
# print(time() - t)
# #%%
# results = []

# cqts = np.stack([analyser.cqt_rough(a) for a in audio])

#%%
# def profile():
#     results = []
#     for c in cqts:
#         c = c/c.max()
#         with torch.no_grad():
#             c = torch.from_numpy(c).unsqueeze(0).to(torch.float)
#             #print(c.dtype, c.shape)
#             result = analyser.note_pred_net(c)
#             results.append(result.detach().numpy().squeeze())
#     return results

# t = time()
# results_1 = profile()
# print(time() - t)

#%%
# start = 0
# length = len(prds) - 1
# end = start + length

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 6))
# ax1.imshow(prds[start:end].T, aspect = 'auto', origin = 'lower')
# ax2.imshow(cqts[start:end].T, aspect = 'auto', origin = 'lower')



# %%
# """ PROFILING """

# import cProfile, pstats
# from wave_generator import gen_polyphonic_wave

# def profile(audio):
#     refresh = True
#     for a in audio:
#         result = analyser.analyse(a.numpy())

# profiler = cProfile.Profile()
# profiler.enable()
# profile(audio)
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('cumtime')
# stats.print_stats()
# %%

# """ testing cqt """


# import torchaudio

# file_path = '/home/cyberblaster/Audio/new_okarina.wav'

# flen = 1024 * 2

# audio, sr = torchaudio.load(file_path)
# audio = audio[0]

# a_len = len(audio) // flen


# audio = audio[:a_len*flen].view(a_len, flen)


# cqt_rough = RoughCQT(sr = sr, 
#                         filter_length=flen, 
#                         filter_ratios=[1,2,2], 
#                         bins_per_octave=36,
#                         n_filters=96)

# cqt1 = np.stack([cqt_rough(a.numpy()) for a in audio])

# cqt_rough = RoughCQT(sr = sr, 
#                         filter_length=flen, 
#                         filter_ratios=[1,1,1], 
#                         bins_per_octave=36,
#                         n_filters=96)

# cqt2 = np.stack([cqt_rough(a.numpy()) for a in audio])

# #%%

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 10))
# ax1.imshow(cqt1.T, aspect = 'auto')
# ax2.imshow(cqt2.T, aspect = 'auto')

# %%
