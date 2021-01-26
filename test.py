import sys
from PyQt5.QtWidgets import *


class MainApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)

        widget = QPushButton('test')

        window = QMainWindow()
        window.setCentralWidget(widget)

        window.show()

        self.window = window


app = MainApp(sys.argv)
app.exec_()

