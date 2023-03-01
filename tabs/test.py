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

        self.upload_button = QPushButton("Upload", self)
        self.upload_button.setFixedWidth(75)
        self.upload_button.clicked.connect(self.upload)

        self.upload_image = QLabel(self)
        self.upload_image.setMaximumSize(int(width / 2 + 1), int(width / 2) + 1)
        self.upload_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.test_button = QPushButton("Test", self)
        self.test_button.setFixedWidth(100)
        self.test_button.clicked.connect(self.test)
  

        self.layout().addWidget(self.upload_button, 0, 0)
        self.layout().addWidget(self.upload_image, 2, 0)
        self.layout().addWidget(self.test_button, 0, 3)

    def upload(self):
        options = QFileDialog.Option.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   "QFileDialog.getOpenFileName()", "", "JSON Files (*.json)",
                                                   options=options)
        if file_name:
            print(file_name)
            self.storage = file_name

            with open("paths.yaml", "w") as f:
                f.write("")

            with open(self.storage, "r") as f:
                data = json.load(f)

            with open("paths.yaml", "w") as f:
                yaml.dump(data, f)
        # self.yaml()
        print("finish line")

    def test(self):
        # load a custom model
        # model = YOLO("/Users/cadenli/Documents/FractureX-Dataset/best.pt")
        # model.conf_thres = 0.427
        # # with open("paths.yaml", "r") as f:
        # #     data = yaml.safe_load(f)
        # print(model.conf_thres)


        output_stream = run(["yolo task=detect mode=val model=/Users/cadenli/Documents/FractureX-Dataset/best.pt data=/Users/cadenli/Documents/FractureX-Dataset/data.yaml conf=0.427"], text=True, capture_output=True, shell=True)
        
        stringData = str(output_stream)

        print(":SLFKJSDL:FDS:FKL:DJKFL:DSF")
        print(stringData)
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
            
                
        image_path = directory_path + "/val" + str(num) + "/PR_curve.png"

        print(image_path)
        pixmap = QPixmap(image_path)
        self.upload_image.setPixmap(pixmap)
        max_size = QSize(int(self.width/2) + 1, int(self.height/2) + 1)
        print(pixmap.width(), pixmap.height())
        print(self.width, self.height)
        print(max_size)
        scaled_pixmap = pixmap.scaled(max_size,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
        self.upload_image.setPixmap(scaled_pixmap)
        self.storage = image_path
