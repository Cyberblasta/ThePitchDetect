#%%
import numpy as np

class EMA:
    def __init__(self, c = 0.99):

        self.x = 0
        self.c = c
        self.curr = 0

    def update(self, n):
        self.x = n

    def refresh(self):
        self.x = 0

    def __call__(self):
        self.curr = self.curr * self.c + self.x * (1 - self.c)
        return self.curr
    
#%%

# class EMA:
    # def __init__(self, c = 0.99):
    #     self.x = 0
    #     self.c = c

    # def update(self, n):
    #     if self.x == 0 or n == 0: ## CHANGED
    #         self.x = n
    #     self.x = self.x*self.c + n*(1-self.c)

    # def refresh(self):
    #     self.x = 0

    # def __call__(self):
    #     return np.round(self.x, 2)