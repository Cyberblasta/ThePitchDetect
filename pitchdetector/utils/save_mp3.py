import numpy as np
from pydub import AudioSegment
import datetime

def save_numpy_to_mp3(audio, filename, 
                      sample_rate = 44100, bitrate = '128k'):

    audio = np.int16(audio.flatten()).tobytes()
    audio_segment = AudioSegment(audio, frame_rate = sample_rate,
                                 sample_width = 2,
                                 channels = 1)
    audio_segment.export(filename, format = 'mp3', bitrate=bitrate)