import copy
import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from torch.cuda import is_available
from ultralytics import YOLO

from base import BaseDetector
from config import CameraConfig, InferenceConfig
from utils import preprocess_yolo_boxes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

inference_config = InferenceConfig()


class YoloDetector(BaseDetector):
    def __init__(
        self,
        model_name: Optional[str] = inference_config.YOLO_MODEL_NAME,
        out_folder: str = "./inference",
    ):
        super().__init__()
        self.model_name = model_name
        self.device = "0" if is_available() else "cpu"
        self.out_folder = Path(out_folder)
        self.out_folder.mkdir(exist_ok=True, parents=True)
        # this is meant to be used/adapted with a classifier on top of
        # yolo detections. skip it for now
        self.classifier = None
        self.load_model()
        self.inference_method = {"image": self.model, "video": self.model.track}

    def load_model(self):
        self.model = YOLO(self.model_name)

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
            classes=inference_config.YOLO_DEFAULT_CLASSES,
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


if __name__ == "__main__":
    detector = YoloDetector()
    config = {"save_to_disk": True, "method": "video"}
    results = detector.detect_camera(camera_config=CameraConfig(), **config)
