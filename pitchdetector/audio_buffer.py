import pyaudio
import sounddevice

from PyQt5.QtCore import QTimer

import numpy as np

import threading

from .config import config

from collections import deque

from math import ceil

pa = pyaudio.PyAudio()

class AudioBuffer():
    def __init__(self):
        self.stream = pa.open(
            format=pyaudio.paInt16, 
            channels=1, 
            rate=44100, 
            input=True, 
            frames_per_buffer=4096,
            #input_device_index=18,
            stream_callback=self.callback)
        
        self.audio_buffer = deque(maxlen=646)
        self.new_frames_count = 0
        self.lock = threading.Lock()

    def callback(self, input, frame_count, time_info, status):
        audio = np.frombuffer(input, dtype=np.int16)
        with self.lock:
            audio = audio/32768
            audio = audio.reshape(1, -1)
            self.audio_buffer.append(audio)
            self.new_frames_count += 1
        return (audio, pyaudio.paContinue)
    
    from PyQt5.QtCore import QTimer

class AudioCaller:
    def __init__(self, widget):
        self.audio_buffer = AudioBuffer()
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_frequency)
        self.timer.start(int(1000 * config.frame_length / config.sample_rate))
        self.widget = widget
        self.refresh = True

        #self.analyzer = Analyzer(length=1024*4, level_threshold=0.1)

    def detect_frequency(self):
        if len(self.audio_buffer.audio_buffer) == 0:
            return
        
        audio = (self.audio_buffer.audio_buffer[-1]).squeeze()#.reshape(1,1,4096)
        #result = self.analyzer.analyse(audio)
        
        self.widget.frequency_detected.emit(audio)#result)

        self.audio_buffer.audio_buffer = []