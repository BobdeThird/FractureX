from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from ultralytics import YOLO

import numpy as np
import cv2


class ViewTab(QWidget):
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
        self.upload_image.setMaximumSize(int(width / 3 + 1), int(width / 3) + 1)

       
        self.process_button = QPushButton("Process", self)
        self.process_button.setFixedWidth(100)
        self.process_button.clicked.connect(self.process)
        self.process_image = QLabel(self)
        self.process_image.setMaximumSize(int(width / 3 + 1), int(width / 3) + 1)


        self.layout().addWidget(self.upload_button, 0, 0)
        self.layout().addWidget(self.upload_image, 2, 0)
        self.layout().addWidget(self.process_button, 0, 3)
        self.layout().addWidget(self.process_image, 2, 3)

    def upload(self):
        options = QFileDialog.Option.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self,
            "QFileDialog.getOpenFileName()", "", "Images (*.png *.xpm *.jpg *jpeg *.bmp);;All Files (*)",
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


    def process(self):
        if self.storage != "":
            model = YOLO("/Users/cadenli/Documents/FractureX-Dataset/best.pt")  # load a custom model
            
            results = model(self.storage)
            print("FINISHED")
            print(results[0].boxes.boxes)
            self.plot_bboxes(cv2.imread(self.storage), results[0].boxes.boxes, score=True)

    def box_label(self, image, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            fontThickness = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=fontThickness)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        lw / 3,
                        txt_color,
                        thickness=fontThickness,
                        lineType=cv2.LINE_AA)
            
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap(qimage)
        max_size = QSize(int(self.width/2) + 1, int(self.height/2) + 1)
        scaled_pixmap = pixmap.scaled(max_size,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
        self.process_image.setPixmap(scaled_pixmap)
        

    def plot_bboxes(self, image, boxes, labels=[], colors=[], score=True, conf=None):
        if labels == []:
            labels = {0: u'fracture'}
        if colors == []:
            colors = [(89, 161, 197)]
        
        for box in boxes:
            if score:
                label = labels[int(box[-1])] + " " + str(round(100 * float(box[-2]),1)) + "%"
            else:
                label = labels[int(box[-1])]
            if conf :
                if box[-2] > conf:
                    color = colors[int(box[-1])]
                    self.box_label(image, box, label, color)
            else:
                color = colors[int(box[-1])]
                self.box_label(image, box, label, color)
                
        if not boxes.numel():
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap(qimage)
            max_size = QSize(int(self.width/2) + 1, int(self.height/2) + 1)
            scaled_pixmap = pixmap.scaled(max_size,
                                            Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation)
            self.process_image.setPixmap(scaled_pixmap)

