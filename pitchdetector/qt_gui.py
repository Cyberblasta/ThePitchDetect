#%%
from PyQt5.QtWidgets import QWidget, QPushButton
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QPalette
from PyQt5.QtCore import pyqtSignal, QTimer, Qt


#%%

from .analyzer import Analyzer

from .config import config
from .utils.ema import EMA


import pickle
import datetime

import numpy as np

#%%

class GUI(QWidget):

    frequency_detected = pyqtSignal(np.ndarray)#Message1)

    def __init__(self):
        super().__init__()

        self.analyser = Analyzer(length=config.frame_length, 
                                 sr = config.sample_rate,
                                 rough_bins_per_octave=config.rough_bins_per_octave,
                                 rough_n_filters=config.rough_n_filters,
                                 precise_bins_per_octave=config.precise_bins_per_octave,
                                 precise_n_filters=config.precise_n_filters,
                                 level_threshold=config.level_threshold,
                                 harm_noise_threshold=config.harm_noise_threshold,
                                 precise_threshold=config.precise_threshold,
                                 uncertainty_threshold=config.uncertainty_threshold,
                                 cached_precise_filters=config.cached_filters)

        self.setFixedSize(800, 800)
        self.setAutoFillBackground(True)
        #self.setStyleSheet("background-color: rgb(28, 38, 35);")
        self.color = QColor(255, 180, 18)
        self.f1 = EMA(c = 0.95)
        self.f2 = EMA(c = 0.95)
        self.notes = [None, None]
        self.error1 = EMA(c = 0.95)
        self.error2 = EMA(c = 0.95)
        self.volume = EMA(c = 0.95)
        self.frequency_detected.connect(self.handle_frequency_detected)

        self.history = []
        
        button = QPushButton('Log', parent=self)
        button.setGeometry(self.width() - 10 - 40, 10, 40, 25)
        button.clicked.connect(self.save_log)

        p = self.palette()
        p.setColor(QPalette.Window, QColor(28, 38, 35))
        self.setPalette(p)

        timer = QTimer(self)
        timer.setInterval(20)
        timer.timeout.connect(self.update_gui)
        timer.start()

    def update_gui(self):
        self.update()

        #self.show()

    def paintEvent(self, event):

        top_note_idx = 0 if self.f1() > self.f2() else 1
        bottom_note_idx = abs(top_note_idx - 1)
        top_note = self.notes[top_note_idx]
        bottom_note = self.notes[bottom_note_idx]
        top_freq = [self.f1, self.f2][top_note_idx]
        bottom_freq = [self.f1, self.f2][bottom_note_idx]

        tick_length = 25
        tick_start = 50
        tick_end = self.width() - tick_start
        label_v_offset = -30
        label_h_offset = 20
        tick_spacing = (self.width() - tick_start * 2) // 8

        def paint_scale(height, painter):
            font_size = 14
            #painter = QPainter(self)
            painter.setPen(QPen(self.color, 3))
            painter.setFont(QFont("Helvetica", font_size))
            labels = ['-200ct', '-150ct', '-100ct', '-50ct', '', '50ct', '100ct', '150ct','200ct']
            painter.drawLine(20, height, self.width() - 20, height)
            for n, i in enumerate(range(tick_start, tick_end + 1, tick_spacing)):
                painter.drawLine(i, height - tick_length // 2, i, height + tick_length // 2)
                painter.drawText(i - label_h_offset, height + label_v_offset, labels[n])

        #### SCALE
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        scale_1_height = self.height() // 7 * 2
        scale_2_heigth = self.height() // 7 * 5
        paint_scale(scale_1_height, painter)
        paint_scale(scale_2_heigth, painter)


        painter.setPen(QPen(QColor(252, 72, 0), 3))#Qt.red, 1.5))
        painter.drawLine(int(self.width() // 2 + tick_spacing/50*self.error1()), 
                         scale_1_height - 20, 
                         int(self.width() // 2 + tick_spacing/50*self.error1()), 
                         scale_1_height + 20)
        
        painter.drawLine(int(self.width() // 2 + tick_spacing/50*self.error2()), 
                         scale_2_heigth - 20, 
                         int(self.width() // 2 + tick_spacing/50*self.error2()), 
                         scale_2_heigth + 20)
        
        ### NOTES
        font_size = 100
        painter.setPen(QPen(QColor(252, 72, 0), 1.5))
        painter.setFont(QFont("Helvetica", font_size))
        painter.drawText(self.width() // 2 - font_size // 2 - 40, 
                         scale_1_height - 70,
                         top_note)
        painter.drawText(self.width() // 2 - font_size // 2 - 40, 
                         scale_2_heigth - 70, 
                         bottom_note)
        
        ### ERROR
        font_size = 20
        painter.setPen(QPen(QColor(240, 122, 52), 1.5))
        painter.setFont(QFont("Helvetica", font_size))
        painter.drawText(#self.width()//2 - font_size + 60, 
                         int(self.width() // 2 + tick_spacing/50*self.error1()) - 10, 
                         scale_1_height + 45,
                         str(int(self.error1())))
        painter.drawText(#self.width()//2 - font_size + 60, 
                         int(self.width() // 2 + tick_spacing/50*self.error2()) - 10,
                         scale_2_heigth + 45,
                         str(int(self.error2())))
        
        ### FREQS
        font_size = 40
        painter.setFont(QFont("Helvetica", font_size, QFont.Bold))
        painter.drawText(self.width()//2 - font_size + 160, scale_1_height - 70,
                         str(np.round(top_freq(), 1)))
        painter.drawText(self.width()//2 - font_size + 160, scale_2_heigth - 70,
                         str(np.round(bottom_freq(), 1)))
        
        # ### LEVELS
        # font_size = 16
        # painter.setFont(QFont("Helvetica", font_size, QFont.Bold))
        # painter.drawText(self.width()//2 - font_size + 60, 90,
        #                  str(np.round(top_level(), 1)))
        # painter.drawText(self.width()//2 - font_size + 60, 235,
        #                  str(np.round(bottom_level(), 1)))
        
        ### VOLUME
        font_size = 16
        painter.drawText(10, 90,
                         str(np.round(self.volume(), 1)))
        
        painter.end()

    def handle_frequency_detected(self, input):
        input = self.analyser.analyse(input)
        self.history.append(input)
        if input.freqs[0] != 0 or input.freqs[1] != 0:
            
            f1, f2 = input.freqs
            notes = input.notes
            errors = input.errors
            
            self.f1.update(f1)
            self.f2.update(f2)

            new_notes = [config.notes_to_track[n] if n != -1 else '' for n in notes]

            if self.notes[0] not in new_notes:
                self.error1.refresh()
            if self.notes[1] not in new_notes:
                self.error2.refresh()

            self.error1.update(errors[0])
            self.error2.update(errors[1])
            
            self.notes = new_notes

            #self.update()

    def save_log(self):
        now = datetime.datetime.now()
        date_time = now.strftime("%d%H%M%S")
        with open(f'{config.log_folder}log_{date_time}.log', 'bw') as f:
            pickle.dump(self.history, f)


# import onnxruntime

# model_path = '/mnt/3660917E7771EA5C/Programming/ONNX-test/pitch_detector.onnx'

# session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# input_name = session.get_inputs()[0].name

# def net(cqt):
#     return session.run(None, {input_name : cqt})[0]






# %%
