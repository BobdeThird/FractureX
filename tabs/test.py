from pathlib import Path
import cv2
import numpy as np
import yaml
import json
import os
import torch
from PIL import Image
import io
import urllib
import requests


from subprocess import run
from ultralytics import YOLO


from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtWebEngineWidgets import *


IMG_FORMATS = ('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'dng')
VID_FORMATS = ('mp4', 'mov', 'avi', 'mkv', 'wmv', 'flv')


LOADERS = ['PIL', 'cv2', 'tf', 'torch', 'skimage']


class TestTab(QWidget):
    def __init__(self, width, height):
        super().__init__()
        self.setLayout(QGridLayout())

        self.width = width
        self.height = height

        self.storage = ""
        self.folder_path = "/Users/cadenli/Documents/FractureX-Dataset"
        self.model_name = "/Users/cadenli/Documents/FractureX-Dataset/best.pt"

        self.upload_button = QPushButton("Upload Dataset", self)
        self.upload_button.setFixedWidth(125)
        self.upload_button.clicked.connect(self.upload)

        self.upload_button_model = QPushButton("Upload Model", self)
        self.upload_button_model.setFixedWidth(125)
        self.upload_button_model.clicked.connect(self.upload_model)        

        self.graph_1 = QLabel(self)
        self.graph_1.setMaximumSize(int(width / 2 + 1), int(height / 2) + 1)
        self.graph_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.graph_2 = QLabel(self)
        self.graph_2.setMaximumSize(int(width / 2 + 1), int(height / 2) + 1)
        self.graph_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.graph_3 = QLabel(self)
        self.graph_3.setMaximumSize(int(width / 2 + 1), int(height / 2) + 1)
        self.graph_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.graph_4 = QLabel(self)
        self.graph_4.setMaximumSize(int(width / 2 + 1), int(height / 2) + 1)
        self.graph_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.test_button = QPushButton("Test", self)
        self.test_button.setFixedWidth(100)
        self.test_button.clicked.connect(self.test)

        self.layout().addWidget(self.upload_button, 0, 0)
        self.layout().addWidget(self.upload_button_model, 0, 1)
        self.layout().addWidget(self.test_button, 3, 1)
        self.layout().addWidget(self.graph_1, 1, 0)
        self.layout().addWidget(self.graph_2, 1, 1)
        self.layout().addWidget(self.graph_3, 2, 0)
        self.layout().addWidget(self.graph_4, 2, 1)

    def upload(self):
        options = QFileDialog.Option.ReadOnly
        self.folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')

        # Do something with the selected folder path
        print('Selected folder:', self.folder_path)

    def upload_model(self):
        options = QFileDialog.Option.ReadOnly
        self.model_name, _ = QFileDialog.getOpenFileName(self,
               "QFileDialog.getOpenFileName()", "", "Models (*.pt *.pth);;All Files (*)",
               options=options)
        

        # Do something with the selected folder path
        print('Selected folder:', self.model_name)

    def test(self):
        # load a custom model
        # model = YOLO("/Users/cadenli/Documents/FractureX-Dataset/best.pt")
        # model.conf_thres = 0.427
        # # with open("paths.yaml", "r") as f:
        # #     data = yaml.safe_load(f)
        # print(model.conf_thres)



        dataset = self.folder_path + "/data.yaml"
        print("SD:FLKDSJF:LSDJKF" + dataset + "   " +  self.model_name)

        # output_stream = run([f"yolo task=detect mode=val model={self.model_name} data={dataset} conf=0.427"], text=True, capture_output=True, shell=True)
        
        # stringData = str(output_stream)
        # print(stringData)

        print(":SLFKJSDL:FDS:FKL:DJKFL:DSF")
        # print(stringData)
        # metrics = model.val(
        #     data="/Users/cadenli/Documents/FractureX-Dataset/data.yaml")
        # print(metrics)


        directory_path = "runs/detect"
        num = 0


        # Loop through the subdirectories in the directory
        for subdir in os.listdir(directory_path):
            # Check if the subdirectory is a directory (not a file)
            if os.path.isdir(os.path.join(directory_path, subdir)):
                if "val" in subdir and subdir != "val":
                    if int((subdir[3:])) > num:
                        num = int((subdir[3:]))
            
        if num == 0:
            num = ""
        
        image_path = directory_path + "/val" + str(num) + "/F1_curve.png"
        pixmap = QPixmap(image_path)
        self.graph_1.setPixmap(pixmap)
        max_size = QSize(int(self.width/2.25) + 1, int(self.height/ 2) + 1)
        scaled_pixmap = pixmap.scaled(max_size,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
        self.graph_1.setPixmap(scaled_pixmap)


        image_path = directory_path + "/val" + str(num) + "/PR_curve.png"
        pixmap = QPixmap(image_path)
        self.graph_2.setPixmap(pixmap)
        max_size = QSize(int(self.width/2.25) + 1, int(self.height/ 2) + 1)
        scaled_pixmap = pixmap.scaled(max_size,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
        self.graph_2.setPixmap(scaled_pixmap)


        image_path = directory_path + "/val" + str(num) + "/P_curve.png"
        pixmap = QPixmap(image_path)
        self.graph_3.setPixmap(pixmap)
        max_size = QSize(int(self.width / 2.25 + 1), int(self.height / 2) + 1)
        scaled_pixmap = pixmap.scaled(max_size,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
        self.graph_3.setPixmap(scaled_pixmap)


        image_path = directory_path + "/val" + str(num) + "/R_curve.png"
        pixmap = QPixmap(image_path)
        self.graph_4.setPixmap(pixmap)
        max_size = QSize(int(self.width / 2.25 + 1), int(self.height / 2) + 1)
        scaled_pixmap = pixmap.scaled(max_size,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
        self.graph_4.setPixmap(scaled_pixmap)

