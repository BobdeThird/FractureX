##### IMPORTS #####
import cv2
import json
import numpy as np
import subprocess
import torch
import os


from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtWebEngineWidgets import *


from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances


import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
##### IMPORTS #####\


##### DATA #####
with open("GRAZPEDWRI-DX/test/_annotations.coco.json") as json_file:
   data = json.load(json_file)
x_train = np.array([])
y_train = np.array([])


path = "GRAZPEDWRI-DX/test/"


for item in data["images"]:
   np.append(x_train, cv2.imread(path + item["file_name"]))


for item in data["annotations"]:
   np.append(y_train, item["bbox"])


x_train1 = np.array(x_train)
y_train1 = np.array(y_train)


register_coco_instances("my_dataset_train", {}, "GRAZPEDWRI-DX/train/_annotations.coco.json", "GRAZPEDWRI-DX/train/")
register_coco_instances("my_dataset_val", {}, "GRAZPEDWRI-DX/valid/_annotations.coco.json", "GRAZPEDWRI-DX/valid/")
register_coco_instances("my_dataset_test", {}, "GRAZPEDWRI-DX/test/_annotations.coco.json", "GRAZPEDWRI-DX/test/")
##### DATA #####\


##### CRY #####
class CocoTrainer(DefaultTrainer):


 @classmethod
 def build_evaluator(cls, cfg, dataset_name, output_folder=None):


   if output_folder is None:
       os.makedirs("coco_eval", exist_ok=True)
       output_folder = "coco_eval"


   return COCOEvaluator(dataset_name, cfg, False, output_folder)
##### CRY #####\


##### TEST TAB #####
class TestTab(QWidget):
   def __init__(self, width, height):
       super().__init__()


       #### PROPERTIES ####
       # self.setLayout(QGridLayout())
       self.setLayout(QGridLayout())
       self.width = width
       self.height = height
       #### PROPERTIES ####\


      
       #### TENSORFLOW WIDGETS ####
       self.web_view = QWebEngineView(self)
       self.web_view.setFixedSize(800, 600)


       self.tensorboard_button = QPushButton("Launch TensorBoard", self)
       self.tensorboard_button.setFixedWidth(200)
       self.tensorboard_button.clicked.connect(self.launch_tensorboard)


       self.train_button = QPushButton("Train", self)
       self.train_button.setFixedWidth(100)
       self.train_button.clicked.connect(self.train_model)
       #### TENSORFLOW WIDGETS ####\


       #### LAYOUT ####
       self.layout().addWidget(self.web_view, 1, 0)
       self.layout().addWidget(self.tensorboard_button, 1, 1)
       self.layout().addWidget(self.train_button, 1, 2)
       #### LAYOUT ####
  
   def train_model():
       ### INITIALIZES MODEL ###
       cfg = get_cfg()
       cfg.MODEL.DEVICE = "cpu"
       cfg.merge_from_file("detectron2parent/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
       cfg.MODEL.WEIGHTS = "cfg_documents/model_final.pth"
       cfg.DATASETS.TRAIN = ("my_dataset_train")
       cfg.DATASETS.VALIDATION = ("my_dataset_val")
       cfg.DATASETS.TEST = ("my_dataset_train")
       cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
       cfg.SOLVER.MAX_ITER = 300
       cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
       cfg.MODEL.ROI_HEADS.NUM_ROIS_TRAINING = 2000
       cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50


       cfg.DATALOADER.NUM_WORKERS = 4
       cfg.SOLVER.IMS_PER_BATCH = 4
       cfg.SOLVER.BASE_LR = 0.001


       cfg.SOLVER.WARMUP_ITERS = 250
       cfg.SOLVER.STEPS = (250, 2000)
       cfg.SOLVER.GAMMA = 0.05


       cfg.TEST.EVAL_PERIOD = 50
       # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
       trainer = CocoTrainer(cfg)
       # trainer = DefaultTrainer(cfg)
       ### INITIALIZES MODEL ###\


       ### TRAINS MODEL ###
       # writer = SummaryWriter()


       # criterion = torch.nn.CrossEntropyLoss()
       # optimizer = torch.optim.SGD(trainer.parameters(), lr=0.01, momentum=0.9)


       trainer.resume_or_load(resume=False)
       print("fishfishfishfishfisfhsifhfihsifhs")
       trainer.train()
       print("deaddeaddeaddeaddeaddeaddeaddeaddeaddead")
       # for i in range(300):
       #     # Forward pass
       #     output = trainer(x_train)
       #     loss = criterion(output, y_train)
          
       #     # Log the loss to TensorBoard
       #     writer.add_scalar('Loss/train', loss, i)


       #     # Backward and optimize
       #     optimizer.zero_grad()
       #     loss.backward()
       #     optimizer.step()
          
       #     if (i+1) % 10 == 0:
       #         print ('Epoch [{}/{}], Loss: {:.4f}'
       #                .format(i+1, 300, loss.item()))
      
       # writer.close()
      
       # for iteration, data in enumerate(trainer.data_loader):
       #     # Get the input and target data
       #     input = data["image"].to("cpu")
       #     target = data["instances"].to("cpu")


       #     # Forward pass of the model
       #     output = trainer(input)
          


       #     # Calculate the loss
       #     loss = criterion(output, target)
          
       #     # Backward pass and optimization
       #     optimizer.zero_grad()
       #     loss.backward()
       #     optimizer.step()
       #     writer.add_scalar('loss', loss, global_step=iteration)
       #     writer.add_scalar('accuracy', accuracy, global_step=iteration)


       ### TRAINS MODEL ###\


   def launch_tensorboard(self):
       # Launch TensorBoard in the background
       subprocess.Popen(["tensorboard", "--logdir=runs", "--host=127.0.0.1", "--style dark"])


       # Wait for TensorBoard to start
       import time
       time.sleep(2)


       # Load TensorBoard web interface in the web view
       self.web_view.load(QUrl("http://127.0.0.1:6006"))
  
   # #### UPLOAD BUTTON ####
   # def upload(self):
   #     options = QFileDialog.Option.ReadOnly
   #     file_name, _ = QFileDialog.getOpenFileName(self,
   #         "QFileDialog.getOpenFileName()", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)",
   #         options=options)
   #     if file_name:
   #         pixmap = QPixmap(file_name)
   #         max_size = QSize(int(self.width / 3), int(self.height / 3))
   #         scaled_pixmap = pixmap.scaled(max_size,
   #                                         Qt.AspectRatioMode.KeepAspectRatio,
   #                                         Qt.TransformationMode.SmoothTransformation)
   #         self.upload_image.setPixmap(scaled_pixmap)
   # #### UPLOAD BUTTON ####\
##### TEST TAB #####\


##### MAIN #####
def main():
   TestTab.train_model()


if __name__ == "__main__":
   main()
##### MAIN #####\



