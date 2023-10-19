#%%

class Message1:
    def __init__(self, d = None):

        self.freqs = [0, 0]
        self.vols = [0, 0]
        self.notes = [-1, -1    ]
        self.probs = [0]
        self.errors = [0, 0]
        self.refresh = False
        self.uncertainty = 0
        
        self.level = 0
        self.noise = 0

        self.wave = None
        self.cqt = None

        if d is not None:
            self.add(d)

    def add(self, d):
        for key in d.keys():
            setattr(self, key, d[key])
        return self

    def __repr__(self):
        return str(self.__dict__)
    
    def __call__(self):
        return self.__dict__

#%%

import numpy as np

class Messages:
    
    def __init__(self):

        self.freqs = []
        self.vols = []
        self.notes = []
        self.probs = []
        self.errors = []
        self.refresh = []
        self.uncertainty = []
        
        self.level = []
        self.noise = []

        self.wave = []
        self.cqt = []

        self.cathegorical = ['refresh']

    def add(self, msg):
        for key in msg().keys():
            getattr(self, key).append(getattr(msg, key))
        return self
    
    def numpyfy(self):
        self.denone_probs()
        for attr in self.__dict__.keys():
            if attr not in self.cathegorical:
                setattr(self, attr, np.array(getattr(self, attr)))

        
        return self
    
    def denone_probs(self):
        size = 0
        for p in self.probs:
            if len(p) != 1:
                size = len(p)
                #dtype = p.dtype
        for n, p in enumerate(self.probs):
            if len(p) == 1:
                self.probs[n] = np.zeros(size)
        self.probs = np.stack(self.probs)
        

#%%

# m = Message1()

# fram  = m.add({'freqs': [300, 400], 'errors': {10, 20}})

# #%%

# msgs = Messages()

# for i in range(1000):
#     freqs = [i*10, i*20]
#     refresh = (i % 3) == 0
#     notes = [i, i+1]
#     msgs.add(m.add({'freqs': freqs, 'refresh': refresh, 'notes': notes}))

# # %%

# %%
