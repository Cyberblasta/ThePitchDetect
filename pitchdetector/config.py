#%%
import yaml

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

with open('config.yaml', 'r') as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)

config = Config(**config_dict)
