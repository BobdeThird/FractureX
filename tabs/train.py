##### IMPORTS #####
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *


from PIL import Image


from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor


from processing import immediate_process
##### IMPORTS #####\


##### TRAIN TAB #####
class TrainTab(QWidget):
   def __init__(self, width, height):
       super().__init__()


       #### PROPERTIES ####
       self.setLayout(QGridLayout())
       self.width = width
       self.height = height
       #### PROPERTIES ####\


       #### VIEWING WIDGETS ####
       self.layout().addWidget(QLabel("This is the training tab"))
       #### VIEWING WIDGETS ####\
##### TRAIN TAB #####\

