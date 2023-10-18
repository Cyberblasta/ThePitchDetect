
#%%

from pitchdetector.main_window import MainWindow

from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()