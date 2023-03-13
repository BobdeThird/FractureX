from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
import onnxruntime
import time

# opt_session = onnxruntime.SessionOptions()
# opt_session.enable_mem_pattern = True
# opt_session.enable_cpu_mem_arena = True
# opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

# model_path = '/Users/cadenli/Documents/FractureX-Dataset/best.onnx'
# EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# start = time.time()
# ort_session = onnxruntime.InferenceSession(model_path, None)

# model_inputs = ort_session.get_inputs()
# input_names = [model_inputs[i].name for i in range(len(model_inputs))]
# input_shape = model_inputs[0].shape


# model_output = ort_session.get_outputs()
# output_names = [model_output[i].name for i in range(len(model_output))]

# print(output_names)

import cv2
import numpy as np
from PIL import Image

# image_path = '/Users/cadenli/Documents/FractureX-Dataset/chicken/0015_0668695173_01_WRI-L1_F008_jpeg.rf.633eabff5333b80c15a41693c999ddb0.jpg'
# image = cv2.imread(image_path)
# image_height, image_width = image.shape[:2]
# print(image_height, image_width)
# Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# size = max(input_shape[2:])

# input_height, input_width = size, size
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# resized = cv2.resize(image_rgb, (input_width, input_height))
# cv2.imshow('bruh', resized)
# # Scale input pixel value to 0 to 1
# input_image = resized / 255.0
# input_image = input_image.transpose(2,0,1)
# input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
# print(input_tensor.shape)

# outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]


# predictions = np.squeeze(outputs).T
# conf_thresold = 0.0
# # Filter out object confidence scores below threshold
# scores = np.max(predictions[:, 4:], axis=1)
# print(predictions)

# predictions = predictions[scores > conf_thresold, :]
# print(predictions)

# scores = scores[scores > conf_thresold]  

# # Get the class with the highest confidence
# class_ids = np.argmax(predictions[:, 4:], axis=1)

# # Get bounding boxes for each object
# boxes = predictions[:, :4]

# #rescale box
# input_shape = np.array([input_width, input_height, input_width, input_height])
# boxes = np.divide(boxes, input_shape, dtype=np.float32)
# boxes *= np.array([image_width, image_height, image_width, image_height])
# boxes = boxes.astype(np.int32)

# print("TIME INFERENCE (Sec):", time.time() - start)

def compute_iou(box, boxes):
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

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    print(sorted_indices)
    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        print(ious)
        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious > iou_threshold)[0]
        

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


# # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
# indices = nms(boxes, scores, 0.4)
# print('indices', indices)


# CLASSES = ['fracture']

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def plot_box(self, boxes, scores, class_ids, CLASSES, indices, image):
        for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
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


# image_draw = image.copy()
# for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
#     bbox = bbox.round().astype(np.int32).tolist()
#     cls_id = int(label)
#     cls = CLASSES[cls_id]
#     color = (0,255,0)
#     cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
#     cv2.putText(image_draw,
#                 f'{cls}:{int(score*100)}%', (bbox[0], bbox[1] - 2),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.60, [225, 255, 255],
#                 thickness=1)
# cv2.imshow('image',image_draw)
# cv2.waitKey(0)
