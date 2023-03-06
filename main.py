import sys


from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

from tabs.view import ViewTab
from tabs.test import TestTab


name = "FractureX"
width = 900
height = 800


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




class MainWindow(QMainWindow):
 def __init__(self):
     dialog = WelcomeWindow()
     dialog.exec()
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
