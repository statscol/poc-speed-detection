import cv2
import numpy as np


def preprocess_yolo_boxes(boxes):
    """preprocess yolov8 ultralytics boxes in format xywh which have x center,y_center width, height"""
    boxes_xywh = [
        [int(x_c - (w / 2)), int(y_c - (h / 2)), w, h] for (x_c, y_c, w, h) in boxes
    ]
    return boxes_xywh


def crop_image(box, image):
    "box: list| array with xyxy positions"
    x_i, y_i, x_f, y_f = box.astype(int)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image[y_i:y_f, x_i:x_f]


class ViewTransformer:
    """taken from Roboflow tutorial
    https://blog.roboflow.com/estimate-speed-computer-vision/
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
