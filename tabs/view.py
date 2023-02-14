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


##### VIEW TAB #####
class ViewTab(QWidget):
   def __init__(self, width, height):
       super().__init__()


       #### PROPERTIES ####
       self.setLayout(QGridLayout())
       self.width = width
       self.height = height
       #### PROPERTIES ####\


       ### UPLOAD STORAGE ####
       self.storage = ""
       ### UPLOAD STORAGE ####\


       #### VIEWING WIDGETS ####
       self.upload_button = QPushButton("Upload", self)
       self.upload_button.setFixedWidth(75)
       self.upload_button.clicked.connect(self.upload)
       self.upload_image = QLabel(self)
       self.upload_image.setMaximumSize(int(width / 3 + 1), int(width / 3) + 1)


       self.process_button = QPushButton("Process", self)
       self.process_button.setFixedWidth(100)
       self.process_button.clicked.connect(self.process)
       self.process_image = QLabel(self)
       self.process_image.setMaximumSize(int(width / 3 + 1), int(width / 3) + 1)
       #### VIEWING WIDGETS ####\


       #### LAYOUT ####
       self.layout().addWidget(self.upload_button, 0, 0)
       self.layout().addWidget(self.upload_image, 2, 0)
       self.layout().addWidget(self.process_button, 0, 3)
       self.layout().addWidget(self.process_image, 2, 3)
       #### LAYOUT ####\


   #### UPLOAD BUTTON ####
   def upload(self):
       options = QFileDialog.Option.ReadOnly
       file_name, _ = QFileDialog.getOpenFileName(self,
           "QFileDialog.getOpenFileName()", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)",
           options=options)
       if file_name:
           pixmap = QPixmap(file_name)
           self.upload_image.setPixmap(pixmap)
           max_size = QSize(int(self.width/2) + 1, int(self.height/2) + 1)
           scaled_pixmap = pixmap.scaled(max_size,
                                           Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)
           self.upload_image.setPixmap(scaled_pixmap)
           self.storage = file_name
   #### UPLOAD BUTTON ####\


   ##### PROCESS BUTTON ####
   def process(self):
       ### INITIALIZES MODEL ###
       cfg = get_cfg()
       print(cfg.OUTPUT_DIR)
       cfg.MODEL.DEVICE = "cpu"
       cfg.merge_from_file("detectron2parent/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
       cfg.MODEL.WEIGHTS = "cfg_documents/model_final.pth"
       cfg.DATASETS.TEST = (cfg.DATASETS.TRAIN, )
       cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
       cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
       predictor = DefaultPredictor(cfg)
       ### INITIALIZES MODEL ###\


       if self.storage != "":
           ndarray = immediate_process(self.storage, predictor)
           img = Image.fromarray(ndarray)
           buf = img.tobytes("raw", "RGBA")
           qim = QImage(buf, img.size[0], img.size[1], QImage.Format.Format_ARGB32)
           pixmap = QPixmap.fromImage(qim)
           max_size = QSize(int(self.width / 2) + 1, int(self.height / 2) + 1)
           scaled_pixmap = pixmap.scaled(max_size,
                                           Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)
           self.process_image.setPixmap(scaled_pixmap)
   # #### PROCESS BUTTON ####\
##### VIEW TAB #####\

