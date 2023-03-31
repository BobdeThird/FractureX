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

        self.width = width
        self.height = height
        self.storage = "" # TODO: change storage into more descriptive (path of models??)
        self.style = 0

        # create a layout for the tab
        self.layout = QGridLayout(self)

        # Create a top layout for the buttons
        self.top_layout = QHBoxLayout()
        self.layout.addLayout(self.top_layout, 0, 0, 1, -1)

        # Create combo boxes and buttons
        self.combo_box_1 = QComboBox()
        self.combo_box_1.addItems(["Individual", "Folder"])
        self.combo_box_1.currentIndexChanged.connect(self.onComboBoxChanged)
        self.combo_box_2 = QComboBox()
        self.combo_box_2.addItems(["YoloV8v1", "YoloV8v2", "YoloV8v3"])
        self.upload_button = QPushButton("Upload and Process")
        self.upload_button.clicked.connect(self.upload)

        # Add the buttons to the top layout
        self.top_layout.addWidget(self.combo_box_1)
        self.top_layout.addWidget(self.upload_button)
        self.top_layout.addWidget(self.combo_box_2)

         # Create a scroll area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        self.upload_image = QLabel(self)
        self.process_image = QLabel(self)

        self.label = QLabel()
        self.label2 = QLabel()

        # Add a stylesheet to the scrollbar
        self.scroll_area.verticalScrollBar().setStyleSheet("QScrollBar:vertical {"
                                                       "    border: none;"
                                                       "    background: #f6f6f6;"
                                                       "    width: 15px;"
                                                       "    margin: 0px 0 0px 0;"
                                                       "}"
                                                       "QScrollBar::handle:vertical {"
                                                       "    background: #a9a9a9;"
                                                       "    border-radius: 7px;"
                                                       "    min-height: 20px;"
                                                       "}"
                                                       "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
                                                       "    background: none;"
                                                       "}"
                                                       "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
                                                       "    background: #dcdcdc;"
                                                       "}")



        # Create a widget to hold the images
        self.images_widget = QWidget(self.scroll_area)
        self.images_layout = QGridLayout(self.images_widget)

        # Set the widget to the scroll area
        self.scroll_area.setWidget(self.images_widget)
        # Add the scroll area to the layout
        self.layout.addWidget(self.scroll_area, 1, 0, -1, -1)


    
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

    # method to make sure that bounding boxes on the same object get removed
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
    
    # TODO: Find out how zip works, as well as reformat the cv2 stuff
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
                        1.00, [225, 255, 255],
                        thickness=2)



    # TODO: onCOmboBoxChange changed so that it works for the models too. change self.style to something more specific like self.currModel and self.currInputType (image / folder)
    def onComboBoxChanged(self, index):
        currentChoice = self.combo_box_1.currentText()
        if currentChoice == "Individual":
            self.style = 0
        else:
            self.style = 1


    def upload(self):
        # self.process_image.clear()
        # self.upload_image.clear()
        # self.label.clear()
        # self.label2.clear()
        # self.images_widget.deleteLater()

        # Create a widget to hold the images
        self.images_widget = QWidget(self.scroll_area)
        self.images_layout = QGridLayout(self.images_widget)

        # Set the widget to the scroll area
        self.scroll_area.setWidget(self.images_widget)
        # Add the scroll area to the layout
        self.layout.addWidget(self.scroll_area, 1, 0, -1, -1)

        print(self.style)
        if self.style == 0:
            options = QFileDialog.Option.ReadOnly
            file_name, _ = QFileDialog.getOpenFileName(self,
                                                       "QFileDialog.getOpenFileName()", "", "Images (*.png *.xpm *.jpg *jpeg *.bmp);;All Files (*)",
                                                       options=options)
            if file_name:
                pixmap = QPixmap(file_name)
                self.upload_image.setPixmap(pixmap)
                max_size = QSize((int(self.width * .88) / 3), int(self.height))
                scaled_pixmap = pixmap.scaled(max_size,
                                              Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)
                
                self.upload_image.setPixmap(scaled_pixmap)
                self.images_layout.addWidget(self.upload_image, 0, 0)
                self.storage = file_name
                scaled_pixmap = self.process()
                self.process_image = QLabel()
                self.process_image.setPixmap(scaled_pixmap)
                self.images_layout.addWidget(self.process_image, 0, 1)

        else:
            options = QFileDialog.Option.ReadOnly
            folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
            image_options = ['.png', '.xpm', '.jpg', 'jpeg', '.bmp']
            row, col = 0, 0
            self.label= QLabel()
            self.label2 = QLabel()

            # Load all images in the folder
            images_folder = folder_path
            for filename in os.listdir(images_folder):
                if filename.endswith(tuple(image_options)):
                    image_path = os.path.join(images_folder, filename)

                    pixmap = QPixmap(image_path)
                    self.label = QLabel()
                    img_width = (self.width * .88) / 3
                    self.label.setPixmap(pixmap.scaled(QSize(int(img_width), 3000), Qt.AspectRatioMode.KeepAspectRatio))
                    self.images_layout.addWidget(self.label, row, col)
                    self.storage = image_path
                    self.process()

                    image_path2 = "arrow.png"
                    pixmap2 = QPixmap(image_path2)
                    self.label2 = QLabel()
                    img_width2 = (self.width * .88) / 3
                    self.label2.setPixmap(pixmap2.scaled(QSize(int(img_width2), 3000), Qt.AspectRatioMode.KeepAspectRatio))
                    self.label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.images_layout.addWidget(self.label2, row, col+1)

                    scaled_pixmap = self.process()
                    self.process_image = QLabel()
                    self.process_image.setPixmap(scaled_pixmap.scaled(QSize(int(img_width2), 3000), Qt.AspectRatioMode.KeepAspectRatio))
                    self.images_layout.addWidget(self.process_image, row, 2)


                    col += 2
                    if col == 2:
                        col = 0
                        row += 1



            
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
            conf_thresold = 0.2
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
            return scaled_pixmap
            #self.images_layout.addWidget(self.process_image, row, 2)

           


