#%%

import numpy as np
from math import log2
#import warnings
from .config import config

NOTE_SYMBOLS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
notes_to_track = config.notes_to_track
C0_FREQUENCY = 16.35


#%%

def get_full_scale():

    notes = []

    for octave in range(7):
        for note_number in range(12):
            note_name = NOTE_SYMBOLS[note_number] + str(octave)
            notes.append(note_name)

    frequencies = []
    
    for note_n in range(len(notes)):
        frequencies.append(C0_FREQUENCY * 2 ** (note_n / 12))

    scale = {}
    for n, note in enumerate(notes):
        scale[note] = frequencies[n]

    return scale

def get_scale(notes = 24, lowest_note = 'C0', return_axis = False):

    full_scale = get_full_scale()

    custom_scale = {}

    add = False
    counter = 0

    for key, value in zip(full_scale.keys(), full_scale.values()):
        if key == lowest_note:
            add = True
        if add:
            custom_scale[key] = value
            counter += 1
        if counter == notes:
            break
    
    return custom_scale

SCALE = get_full_scale()

def get_axis(bins_per_octave = 48, upscaling_factor = 5, lowest_note = 'C0', note_n = 24,
             add_freqs = False):
    
    assert bins_per_octave % 12 == 0, 'bins_per_octave must be divisable by 12.'
    
    bins_per_note = bins_per_octave // 12 * upscaling_factor

    notes = list(SCALE.keys())
    freqs = list(SCALE.values())

    axis = {}
    add = False
    counter = 0

    for n, (note, freq) in enumerate(zip(notes, freqs)):
        if note == lowest_note:
            add = True
        if add:
            if add_freqs:
                axis[counter * bins_per_note] = note + '/' + str(round(freq, 2))
            else:
                axis[counter * bins_per_note] = note
            counter += 1
        if counter == note_n:
            break

    return axis

def note_to_note_number(note):
    for nn, n in enumerate(SCALE.keys()):
        if n == note:
            return nn
    raise Exception(f'No such note {note}.')
    #warnings.warn('No such note.')

def note_to_freq(note):
    return SCALE[note]

def note_to_cents(note):
    freq = SCALE[note]
    cents = 1200 * log2(freq / SCALE['C0'])
    return cents

def notes_to_cents(notes):
    return [note_to_cents(note) for note in notes]

def cents_to_freq(cents):
    min_freq = SCALE['C0']
    freq = min_freq * 2 ** (cents / 1200)
    return freq

def freq_to_cents(freq):
    min_freq = SCALE['C0']
    cents = 1200 * log2(freq / min_freq + 1e-7)
    return cents

def note_number_to_note(number):
    return list(SCALE)[number]

def freq_to_closest_note(freq : float, error_measure : str = None):
    ''' Error measure : 'ct' - cents, or 'hz' - herz. '''
    freqs = np.array([*get_full_scale().values()])
    note_n = (np.abs(freqs - freq).argmin())
    note = list(SCALE.keys())[note_n]
    if error_measure == 'hz':
        return note, SCALE[note] - freq
    elif error_measure == 'ct':
        return note, 1200 * log2(SCALE[note] / freq)
    else:
        return note

def add_margin(notes_to_track, margin, symmetric = False, upper_cent_range = 50, 
               upper_margin = False, lower_margin = False):

    if symmetric:
        print('Symmetric not implemented. Using upper cent margin = 50.')
    else:
        cents_to_track = [note_to_cents(n) + upper_cent_range for n in notes_to_track]

    if lower_margin:
        low_margin_note = note_number_to_note(note_to_note_number(notes_to_track[0]) - margin)
        notes_to_track = [low_margin_note] + notes_to_track
    if upper_margin:
        high_margin_note = note_number_to_note(note_to_note_number(notes_to_track[-1]) + margin)
        notes_to_track = notes_to_track + [high_margin_note]
 
    notes_w_margin = notes_to_track[:]
    
    higher_cents = [note_to_cents(notes_w_margin[-1])] if upper_margin else []
    lower_cents = [note_to_cents(notes_w_margin[0])] if lower_margin else []
    cents_w_margin = lower_cents + cents_to_track + higher_cents

    return notes_w_margin, np.array(cents_w_margin)


def closest_up(cents, cents_w_margin):
    cents = np.tile(cents, (len(cents_w_margin), 1)).astype(np.float32).T
    cents = cents_w_margin.reshape(1, -1) - cents
    cents[cents < 0] = np.inf    
    closest_notes = np.argmin(cents, axis = 1)
    closest_notes[cents.min(-1) == np.inf] = len(cents_w_margin) -1
    return closest_notes

def closest_down(cents, cents_w_margin):
    cents = np.tile(cents, (len(cents_w_margin), 1)).astype(np.float32).T
    cents = cents - cents_w_margin.reshape(1, -1)
    cents[cents < 0] = np.inf    
    closest_notes = np.argmin(cents, axis = 1)
    return closest_notes


# cents = np.array([1000,1200,700,800,1100])
# cents_w_margin = np.array([820, 990, 1115])

# closest_notes = closest_down(cents, cents_w_margin)
# print(closest_notes)



def cents_to_freqs(cents):
    return SCALE['C0'] * 2 ** (cents / 1200)

def gen_dataset_labels(notes_to_track, n_labels, margin = 2, 
                       lower_margin = True, upper_margin = True,
                       close_direction = 'up'):
    """ returns: cents, freqs, note_names, closest_notes """

    notes_w_margin, cents_w_margin = add_margin(notes_to_track, margin = margin,
                                                upper_margin=upper_margin,
                                                lower_margin=lower_margin)


    cents = np.random.randint(min(cents_w_margin), max(cents_w_margin), n_labels)

    if close_direction == 'up':
        closest_notes = closest_up(cents, cents_w_margin)

    if close_direction == 'down':
        closest_notes = closest_down(cents, cents_w_margin)
    # if lower_margin:
    #     closest_notes += 1

    # print('cents:', cents)
    # print('closest_notes:', closest_notes)
    note_names = np.array(notes_w_margin)[closest_notes]
    note_names[note_names == 'C4'] = 'D4'
    note_names[note_names == 'G5'] = 'F5'

    freqs = cents_to_freqs(cents)
    closest_notes[closest_notes == 0] = 1
    #closest_notes[closest_notes == len(notes_w_margin)] = 7
    closest_notes -= 1

    return cents, freqs, note_names, closest_notes


#%%

""" TESTING """

# cents, freqs, note_names, closest_notes = gen_dataset_labels(notes_to_track, 1000)

# #%%

# import random
# r = random.randint(0, len(cents))
# print('cents:', cents[r])
# print('freq:', freqs[r])
# print('absolute_note_number', closest_notes[r])
# print('note name:', note_names[r])
# print('freq:', note_to_freq(note_names[r]))
# print('cents:', note_to_cents(note_names[r]))
# print('')

# notes_w_margin, cents_w_margin  = add_margin(notes_to_track, margin = 2)
# for c, n in zip(cents_w_margin, notes_w_margin):
#     print(c, n, note_to_freq(n))
# #%%
# notes_w_margin, cents_w_margin  = add_margin(notes_to_track[:-2], margin = 2, lower_margin=True)
# for c, n in zip(cents_w_margin, notes_w_margin):
#     print(c, n, note_to_freq(n))

# #%%
# notes_w_margin, cents_w_margin  = add_margin(notes_to_track[2:], margin = 2, upper_margin=True)
# for c, n in zip(cents_w_margin, notes_w_margin):
#     print(c, n, note_to_freq(n))