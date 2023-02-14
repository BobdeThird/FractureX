##### IMPORTS #####
import sys


from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

from view import ViewTab
from train import TrainTab
from test import TestTab
##### IMPORTS #####\
##### CONSTANTS #####
name = "FractureX"
width = 1500
height = 900
##### CONSTANTS #####\




##### WELCOME WINDOW #####
class WelcomeWindow(QDialog):
  def __init__(self, parent=None):
      super().__init__(parent)




      self.setWindowTitle("Welcome")
      self.setFixedSize(QSize(400, 200))




      dialogButton = QDialogButtonBox.StandardButton.Ok




      self.dialogButton = QDialogButtonBox(dialogButton)
      self.dialogButton.accepted.connect(self.accept)
      self.dialogButton.rejected.connect(self.reject)




      self.layout = QVBoxLayout()




      label = QLabel("Welcome to FractureX!")
      label.setAlignment(Qt.AlignmentFlag.AlignCenter)
      self.layout.addWidget(label)
      self.layout.addWidget(self.dialogButton)
      self.setLayout(self.layout)
##### WELCOME WINDOW #####\


##### MAIN WINDOW #####
class MainWindow(QMainWindow):
  #### INITIALIZATION ####
  def __init__(self):
      #### WELCOME WINDOW ####
      dialog = WelcomeWindow()
      dialog.exec()
      #### WELCOME WINDOW ####\


      super(MainWindow, self).__init__()


      #### PROPERTIES ####
      self.setWindowTitle(name)
      # self.set
      self.setFixedSize(QSize(width, height))
      #### PROPERTIES ####\


      #### TABS ####
      tabs = QTabWidget()
      tabs.addTab(ViewTab(width, height), "View")
      tabs.addTab(TrainTab(width, height), "Train")
      tabs.addTab(TestTab(width, height), "Test")
      #### TABS ####\
    
      self.setCentralWidget(tabs)
  #### INITIALIZATION ####\
##### MAIN WINDOW #####


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
