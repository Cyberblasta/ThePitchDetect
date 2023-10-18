from .audio_buffer import AudioCaller

from PyQt5.QtWidgets import QMainWindow, QPushButton

import pickle
import datetime

import numpy as np

from .config import config
from .utils.save_mp3 import save_numpy_to_mp3

from .qt_gui import GUI

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My App")
        self.gui = GUI()
        self.setCentralWidget(self.gui)
        #self.audio_buffer = AudioBuffer()
        #self.audio_analyzer = AudioAnalyzer(self.audio_buffer, self.widget)
        self.audio_caller = AudioCaller(self.gui)

        button = QPushButton('Log', parent=self)
        button.setGeometry(self.width() - 10 - 40, 10, 40, 25)
        button.clicked.connect(self.save_log)

    def save_log(self):
        now = datetime.datetime.now()
        date_time = now.strftime("%d%H%M%S")

        with open(f'{config.log_folder}log_{date_time}.log', 'bw') as f:
            pickle.dump(self.history, f)
        
        audio = np.stack(self.audio_caller.audio_buffer.audio_buffer)
        save_numpy_to_mp3(audio, f'{config.log_folder}{date_time}.mp3')