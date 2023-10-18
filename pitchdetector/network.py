#%%

import onnxruntime
from .config import config
from pathlib import Path

folder = Path('pitchdetector/models/')

model_path = folder / config.model_path

session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name

def net(cqt):
    return session.run(None, {input_name : cqt})[0]
# %%
