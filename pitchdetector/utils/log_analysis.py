#%%

import numpy as np

import matplotlib.pyplot as plt

import pickle

import datetime

with open('logs/log_16214035.log', 'rb') as f:
    data = pickle.load(f)

#%%

from message import Messages

msgs = Messages()

for d in data:
    msgs.add(d)

msgs.numpyfy()


# %%

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize = (10, 7))

ax1.imshow(msgs.probs.T, aspect = 'auto')
ax2.plot(msgs.uncertainty)
ax2.margins(x=0)
ax3.plot(msgs.freqs[:, 0])
ax3.plot(msgs.freqs[:, 1])
ax3.margins(x=0)
ax4.plot(msgs.level)
ax4.plot(np.ones(len(msgs.freqs)) * 0.3, linewidth = 1, color = 'red')
ax4.plot(msgs.noise / msgs.noise.max() * msgs.level.max())
ax4.margins(x=0)
ax5.plot(msgs.refresh)
ax5.margins(x=0)
# %%
