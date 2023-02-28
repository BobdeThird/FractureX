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


       self.test_button = QPushButton("Test", self)
       self.test_button.setFixedWidth(100)
       self.test_button.clicked.connect(self.test)


       self.layout().addWidget(self.upload_button, 2, 0)
       self.layout().addWidget(self.test_button, 2, 3)
  
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
       model = YOLO("best.pt")  # load a custom model
       # with open("paths.yaml", "r") as f:
       #     data = yaml.safe_load(f)
      
       metrics = model.val(data="/Users/jasonzhang/Documents/VisualStudioCode/FractureX/datasets/GRAPE/data.yaml")  # no arguments needed, dataset and settings remembered
       print(metrics)
  
   # def yaml(self):
   #     # with open("paths.yaml", "r") as f:
   #     #     original_data = yaml.safe_load(f)
   #     with open("paths.yaml") as file:
   #         original_data = yaml.load(file, Loader=yaml.FullLoader)
   #     # print(original_data)
   #     print("we're here")


   #     image_paths = []
   #     boxes = []
   #     for item in original_data["images"]:
   #         image_paths.append(item["file_name"])
   #     for item in original_data["annotations"]:
   #         boxes.append(item["bbox"])
      
   #     # image_paths = [item["path"] for item in original_data if "path" in item]
   #     # boxes = [item["box"] for item in original_data if "box" in item]
   #     print(image_paths)
   #     print(boxes)


   #     # yolo_boxes = []
   #     # for box in boxes:
   #     #     # box is assumed to be in the format [xmin, ymin, xmax, ymax]
   #     #     x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
   #     #     yolo_box = [x, y, w, h]
   #     #     yolo_boxes.append(yolo_box)


   #     # Write YOLO format data to file
   #     with open("paths.yaml", "w") as f:
   #         for image_path, box_list in zip(image_paths, boxes):
   #             box_str = " ".join(str(coord) for coord in box_list)
   #             print(box_str)
   #             f.write(f"{image_path}: {box_str}\n")



