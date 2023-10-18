#%%

import numpy as np

from .utils.create_filters import create_filters
from .notes_to_freq import (add_margin, notes_to_track, freq_to_cents,
                           cents_to_freq, note_to_note_number)


class CachedFilters:
    def __init__(self, length, notes_to_track, filters_per_st = 40,
                 filters_per_octave = 36, n_filters = 9):
        
        self.filters_per_octave = filters_per_octave
        self.n_filters = n_filters

        notes, cents = add_margin(notes_to_track, 4, upper_margin=True, lower_margin=True)
        notes_n = note_to_note_number(notes[-1]) - note_to_note_number(notes[0])

        self.min_cents, self.max_cents = cents[0], cents[-1]

        self.cents = np.linspace(self.min_cents, self.max_cents,
                                 filters_per_st * notes_n, 
                                 dtype=np.int32)

        self.freqs = np.array([cents_to_freq(c) for c in self.cents])

        self.filters = create_filters(length, self.freqs)

        self.cents_margin = (self.n_filters - 1) // 2 * 1200 / filters_per_octave

        self.cent_index = np.empty(self.cents[-1] - self.cents[0], dtype=np.int32)

        id = 0
        for n, c in enumerate(range(self.cents[0]-1, self.cents[-1]-1)):
            if c > self.cents[id]:
                id += 1
            self.cent_index[n] = id

    def get_filter_cents(self, freq):

        cents = freq_to_cents(freq)

        min_filter_cents = cents - self.cents_margin
        max_filter_cents = cents + self.cents_margin

        filter_cents = np.linspace(min_filter_cents, max_filter_cents, self.n_filters,
                                   dtype = np.int32)

        return filter_cents

    def get_filters(self, freq):

        filters_cents = self.get_filter_cents(freq)

        if np.any(filters_cents - self.min_cents.astype(np.int32) > (len(self.cent_index) - 1)):
            print(filters_cents)
            print(freq)
        filters_ids = self.cent_index[filters_cents - self.min_cents.astype(np.int32)]

        filters_freqs = self.freqs[filters_ids]

        filters = self.filters[filters_ids]

        return filters, filters_freqs

#%%

# cached_filters = CachedFilters(1024, notes_to_track)

# filters, filter_freqs = cached_filters.get_filters(600)

# #%%

# from math import log2

# filters_per_octave = 36
# n_filters = 9
# cents_per_filter = 1200 / filters_per_octave

# freq = 446

# cents = freq_to_cents(freq)
# print('cents:', cents)

# min_filter_cents = cents - (n_filters - 1) // 2 * cents_per_filter
# max_filter_cents = cents + (n_filters - 1) // 2 * cents_per_filter
# filter_cents = np.linspace(min_filter_cents, max_filter_cents, n_filters)

# %%
