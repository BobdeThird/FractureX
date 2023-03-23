import sys


from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

from tabs.view import ViewTab
from tabs.test import TestTab


name = "FractureX"
width = 900
height = 800

class MainWindow(QMainWindow):
 def __init__(self):
     super(MainWindow, self).__init__()


     self.setWindowTitle(name)
     self.setFixedSize(QSize(width, height))


     tabs = QTabWidget()
     tabs.addTab(ViewTab(width, height), "View")
     tabs.addTab(TestTab(width, height), "Test")
  
     self.setCentralWidget(tabs)
##### MAIN WINDOW #####


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
