from .audio_buffer import AudioCaller

from PyQt5.QtWidgets import QMainWindow

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