import copy
import logging
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from torch.cuda import is_available
from ultralytics import YOLO

from base import BaseDetector
from classifier import CarClassifier, FaceGenderage
from config import DetectorInferenceConfig

# from config import CameraConfig
from utils import preprocess_yolo_boxes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

inference_config = DetectorInferenceConfig()


class YoloDetector(BaseDetector):
    def __init__(
        self,
        model_name: Optional[str] = inference_config.YOLO_MODEL_NAME,
        classifier_path: Optional[str] = inference_config.CLASSIFIER_PATH,
        detector_classes: List[int] = inference_config.YOLO_DEFAULT_CLASSES,
        out_folder: str = "./inference",
    ):
        super().__init__()
        self.model_name = model_name
        self.device = "0" if is_available() else "cpu"
        self.out_folder = Path(out_folder)
        self.detector_classes = detector_classes
        self.out_folder.mkdir(exist_ok=True, parents=True)
        # this is meant to be used/adapted with a classifier on top of
        # yolo detections. skip it for now
        self.classifier_path = classifier_path
        self.load_model()
        self.inference_method = {"image": self.model, "video": self.model.track}

    def load_model(self):
        self.model = YOLO(self.model_name)
        self.classifier = (
            CarClassifier(self.classifier_path)
            if self.classifier_path is not None
            else self.classifier_path
        )

    def detect_image(
        self,
        image: Union[str, np.ndarray],
        save_to_disk: bool = False,
        save_crop: bool = False,
        method: str = "image",
    ):
        """detect persons in an image provided using the @param image, can be a file path or a np.ndarray(cv2 image)"""
        image = cv2.imread(image) if isinstance(image, str) else image
        frame = copy.deepcopy(image)
        detections = self.inference_method[method](
            source=frame,
            device=self.device,
            persist=method == "video",
            verbose=False,
            classes=self.detector_classes,
            save_dir=str(self.out_folder),
            project=str(self.out_folder),
        )

        boxes, names = detections[0].boxes, detections[0].names
        # account for frames where there are no detections
        if boxes is not None:
            boxes_id = (
                None
                if boxes.id is None
                else {
                    int(x.id.cpu().item()): names[int(x.cls.cpu().item())]
                    for x in boxes
                }
            )
            boxes = preprocess_yolo_boxes(boxes.xywh.cpu().numpy().astype(int))
            frame = self.render_boxes(
                image,
                boxes,
                ids=boxes_id,
                save_to_disk=save_to_disk,
                save_crop=save_crop,
                face=False,
            )
        return frame

    def process_videoframe(self, frame, **kwargs):
        """Process videoframes with yolo-ultralytics default object tracking algorithm"""
        return self.detect_image(frame, **kwargs)


class OpenCVFaceDetector(BaseDetector):
    def __init__(self, model_name: str = "", out_folder: str = "./inference"):
        super().__init__()
        self.model_name = model_name
        self.out_folder = Path(out_folder)
        self.classifier = FaceGenderage()
        self.load_model()

    def load_model(self):
        self.model = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self.classifier.prepare()

    def detect_image(self, image: Union[str, np.ndarray], save_to_disk: bool = False):
        out_path = self.get_outpath(image)
        image = cv2.imread(image) if isinstance(image, str) else image
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
        faces = self.model.detectMultiScale(gray, minSize=(30, 30))
        # ids=None as we haven't implemented tracking algorithms for OpenCV based detectors
        image = self.render_boxes(image, faces, ids=None)
        if save_to_disk:
            # save to disk using a random id
            cv2.imwrite(out_path, image)
            self.logger.info(f"Saved detections to {out_path}")
        return image

    def process_videoframe(self, frame, **kwargs):
        return self.detect_image(frame, **kwargs)


class OpenCVPersonDetector(BaseDetector):
    def __init__(self, model_name: str = "", out_folder: str = "./inference"):
        super().__init__()
        self.model_name = model_name
        self.out_folder = Path(out_folder)
        self.classifier = FaceGenderage()
        self.load_model()

    def load_model(self):
        self.model = cv2.HOGDescriptor()
        self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.classifier.prepare()

    def detect_image(self, image: Union[str, np.ndarray], save_to_disk=False):
        """detect persons in an image provided using the @param image, can be a file path or a np.ndarray(cv2 image)"""
        out_path = self.get_outpath(image)
        image = cv2.imread(image) if isinstance(image, str) else image
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)
        # returns the bounding boxes for the detected objects
        boxes, _ = self.model.detectMultiScale(gray, winStride=(8, 8))
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        image = self.render_boxes(image, boxes, ids=None)
        if save_to_disk:
            # save to disk using a random id
            cv2.imwrite(out_path, image)
            self.logger.info(f"Saved detections to {out_path}")
        return image

    def process_videoframe(self, frame, **kwargs):
        return self.detect_image(frame, **kwargs)


if __name__ == "__main__":
    detector = YoloDetector(
        classifier_path=inference_config.CLASSIFIER_PATH,
        model_name=inference_config.YOLO_MODEL_NAME,
    )
    config = {"save_to_disk": False, "method": "video"}
    results = detector.detect_video("./data/video_test.mp4", **config)
    # results = detector.detect_camera(camera_config=CameraConfig(), **config)
