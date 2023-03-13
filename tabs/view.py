from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
# from processing import *
from ultralytics import YOLO
import onnxruntime

import numpy as np
import cv2
from PIL import Image
import os


class ViewTab(QWidget):
    def __init__(self, width, height):
        super().__init__()

        self.setLayout(QGridLayout())
        self.width = width
        self.height = height
        self.storage = ""
        self.style = 0

        self.upload_button = QPushButton("Upload and Process", self)
        self.upload_button.setFixedWidth(150)
        self.upload_button.clicked.connect(self.upload)
        # self.upload_button.setStyleSheet("background-color: rgb(47, 80, 97);")
        self.upload_image = QLabel(self)
        # self.upload_image.setStyleSheet("background-color: rgb(91, 91 91);")
        self.upload_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.upload_image.setMaximumSize(
            int(width / 3 + 1), int(width / 3) + 1)

        self.upload_image_list = [self.upload_image]
        # self.scroll_area = QScrollArea()
        # self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        # self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        # self.scroll_area.setWidgetResizable(True)
        # self.scroll_widget = QWidget()
        # self.scroll_widget.setLayout(QVBoxLayout())
        # self.scroll_widget.layout().addWidget(self.scroll_area)

        self.process_image = QLabel(self)
        # self.process_image.setStyleSheet("background-color: rgb(91, 91 91);")
        self.process_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.process_image.setMaximumSize(
            int(width / 3 + 1), int(width / 3) + 1)
        
        self.process_image_list = [self.process_image]

        self.upload_style = QComboBox(self)
        self.upload_style.addItem("Individual")
        self.upload_style.addItem("Folder")
        # self.upload_style.setStyleSheet("background-color: rgb(47, 80, 97);")
        self.upload_style.setFixedWidth(100)
        self.upload_style.currentIndexChanged.connect(self.onComboBoxChanged)

        self.upload_model = QComboBox(self)
        self.upload_model.addItem("YoloV8v1")
        self.upload_model.addItem("YoloV8v2")
        self.upload_model.addItem("YoloV8v3")
        self.upload_model.addItem("YoloV5")
        self.upload_model.addItem("detectron2v1")
        # self.upload_style.setStyleSheet("background-color: rgb(47, 80, 97);")
        self.upload_model.setFixedWidth(100)

        self.layout().addWidget(self.upload_button, 0, 1)
        self.layout().addWidget(self.upload_style, 0, 0)
        self.layout().addWidget(self.upload_model, 0, 2)
        self.layout().addWidget(self.upload_image, 2, 0)
        # self.layout().addWidget(self.scroll_widget, 2, 0)
        self.layout().addWidget(self.process_image, 2, 3)
    
    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        print(sorted_indices)
        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            print(ious)
            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious > iou_threshold)[0]
            

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes
    
    def plot_box(self, boxes, scores, class_ids, CLASSES, indices, image):
        for (bbox, score, label) in zip(self.xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
            bbox = bbox.round().astype(np.int32).tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = (0,255,0)
            cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
            cv2.putText(image,
                        f'{cls}:{int(score*100)}%', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60, [225, 255, 255],
                        thickness=1)

            


    def onComboBoxChanged(self, index):
        currentChoice = self.upload_style.currentText()
        if currentChoice == "Individual":
            self.style = 0
        else:
            self.style = 1

    def upload(self):
        if self.style == 0:
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
                self.process()
        else:
            options = QFileDialog.Option.ReadOnly
            folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
            image_options = ['.png', '.xpm', '.jpg', 'jpeg', '.bmp']
            numImages = len(os.listdir(folder_path)) 
            scaled_pixmap = 0

            for filename in os.listdir(folder_path):
                f = os.path.join(folder_path, filename)
                # checking if it is a file
                if any(i in f for i in image_options):
                    pixmap = QPixmap(f)
                    curr_upload_image = QLabel(self)
                    # self.process_image.setStyleSheet("background-color: rgb(91, 91 91);")
                    curr_upload_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    curr_upload_image.setMaximumSize(
                        int(self.width / 3 + 1), int(self.width / 3) + 1)
                    
                    curr_upload_image.setPixmap(pixmap)
                    max_size = QSize(int(self.width/2) + 1, int(self.height/2) + 1)
                    scaled_pixmap = pixmap.scaled(max_size,
                                                Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation)
                    curr_upload_image.setPixmap(scaled_pixmap)
                    self.storage = f
                    print(self.storage)
                    self.process()


            
            # options = QFileDialog.Options()
            # options = QFileDialog.Option.ReadOnly
            # options = QFileDialog.Option.ShowDirsOnly
            # folder_path = QFileDialog.getExistingDirectory(
            #     self, "QFileDialog.getExistingDirectory()", "",
            #     options=options
            # )
            # if folder_path:
            #     for i in reversed(range(self.layout.count())):
            #         self.layout.itemAt(i).widget().setParent(None)

            #     for filename in os.listdir(folder_path):
            #         label = QLabel()

            #         pixmap = QPixmap(os.path.join(folder_path, filename))

            #         label.setPixmap(pixmap)
            #         self.layout.addWidget(label)
            #     self.scroll.setWidget(self.widget)


    def process(self):
        opt_session = onnxruntime.SessionOptions()
        opt_session.enable_mem_pattern = True
        opt_session.enable_cpu_mem_arena = True
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        model_path = '/Users/cadenli/Documents/FractureX-Dataset/best.onnx'
        EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        ort_session = onnxruntime.InferenceSession(model_path, None)

        model_inputs = ort_session.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        input_shape = model_inputs[0].shape


        model_output = ort_session.get_outputs()
        output_names = [model_output[i].name for i in range(len(model_output))]

        image_options = ['.png', '.xpm', '.jpg', 'jpeg', '.bmp']

        if any(i in self.storage for i in image_options):
            image_path = self.storage
            print("SDFL:SDKFJSDLKF",image_path)
            image = cv2.imread(image_path)
            
            image_height, image_width = image.shape[:2]
            print(image_height, image_width)
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            size = max(input_shape[2:])
            
            input_height, input_width = size, size
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(image_rgb, (input_width, input_height))
            # Scale input pixel value to 0 to 1
            input_image = resized / 255.0
            input_image = input_image.transpose(2,0,1)
            input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)

            outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]

            predictions = np.squeeze(outputs).T
            conf_thresold = 0.42
            # Filter out object confidence scores below threshold
            scores = np.max(predictions[:, 4:], axis=1)
            #print(predictions)

            predictions = predictions[scores > conf_thresold, :]
            #print(predictions)

            scores = scores[scores > conf_thresold]  

            # Get the class with the highest confidence
            class_ids = np.argmax(predictions[:, 4:], axis=1)

            # Get bounding boxes for each object
            boxes = predictions[:, :4]

            #rescale box
            input_shape = np.array([input_width, input_height, input_width, input_height])
            boxes = np.divide(boxes, input_shape, dtype=np.float32)
            boxes *= np.array([image_width, image_height, image_width, image_height])
            boxes = boxes.astype(np.int32)

            indices = self.nms(boxes, scores, 0.4)

            CLASSES = ['fracture']

            self.plot_box(boxes, scores, class_ids, CLASSES, indices, image)

            height, width, channel = image.shape
            bytes_per_line = 3 * width
            qimage = QImage(image.data, width, height,
                            bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap(qimage)
            max_size = QSize(int(self.width/2) + 1, int(self.height/2) + 1)
            scaled_pixmap = pixmap.scaled(max_size,
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.process_image.setPixmap(scaled_pixmap)
        elif self.storage != "" and self.style == 1:
            numImages = len(os.listdir(self.storage)) 
            scaled_pixmap = 0
            for filename in os.listdir(self.storage):
                f = os.path.join(self.storage, filename)
                # checking if it is a file
                if any(i in f for i in image_options):
                    image_path = f
                    image = cv2.imread(image_path)
                    image_height, image_width = image.shape[:2]
                    #print(image_height, image_width)
                    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    size = max(input_shape[2:])
                    
                    input_height, input_width = size, size
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(image_rgb, (input_width, input_height))
                    # Scale input pixel value to 0 to 1
                    input_image = resized / 255.0
                    input_image = input_image.transpose(2,0,1)
                    input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)

                    outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]

                    predictions = np.squeeze(outputs).T
                    conf_thresold = 0.42
                    # Filter out object confidence scores below threshold
                    scores = np.max(predictions[:, 4:], axis=1)
                    #print(predictions)

                    predictions = predictions[scores > conf_thresold, :]
                    #print(predictions)

                    scores = scores[scores > conf_thresold]  

                    # Get the class with the highest confidence
                    class_ids = np.argmax(predictions[:, 4:], axis=1)

                    # Get bounding boxes for each object
                    boxes = predictions[:, :4]

                    #rescale box
                    input_shape = np.array([input_width, input_height, input_width, input_height])
                    boxes = np.divide(boxes, input_shape, dtype=np.float32)
                    boxes *= np.array([image_width, image_height, image_width, image_height])
                    boxes = boxes.astype(np.int32)

                    indices = self.nms(boxes, scores, 0.4)

                    CLASSES = ['fracture']

                    self.plot_box(boxes, scores, class_ids, CLASSES, indices, image)

                    height, width, channel = image.shape
                    bytes_per_line = 3 * width
                    qimage = QImage(image.data, width, height,
                                    bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap(qimage)
                    max_size = QSize(int(self.width/(numImages + 1)) + 1, int(self.height/(numImages + 1)) + 1)
                    scaled_pixmap = pixmap.scaled(max_size,
                                                Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation)
                    
                    self.process_image.setPixmap(scaled_pixmap)        


