import os
from typing import Dict, List

import cv2
import numpy as np
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class CameraConfig(BaseSettings):
    URL: str = os.getenv("CAMERA_URL", "")
    USERNAME: str = os.getenv("CAMERA_USER", "")
    PASSWORD: str = os.getenv("CAMERA_PASS", "")


class InferenceConfig(BaseSettings):
    YOLO_MODEL_NAME: str = "yolov8s.pt"
    YOLO_DEFAULT_CLASSES: List = [
        0,
        2,
        3,
        5,
        7,
        15,
        16,
    ]  # person,car,motorcycle,bicycle,bus,truck,cat,dog-0,2,3,5,7,15,16
    OFFSET_BOX_CV2: int = 4  # pixels
    FRAMES_UPDATE_ON_VIDEO: int = 30
    CV2_DEFAULT_TEXT_ARGS: Dict = {
        "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
        "fontScale": 1,
        "color": (255, 255, 255),
        "thickness": 2,
    }
    CV2_DEFAULT_TEXT_BOX_ARGS: Dict = {
        "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
        "fontScale": 1,
        "thickness": 2,
    }


class SpeedEstimationConfig(BaseSettings):
    DEFAULT_CONF_SCORE: float = 0.3
    DEFAULT_IOU_THRESHOLD: float = 0.5
    ROI: np.ndarray = np.array([[691, 163], [862, 154], [1026, 629], [795, 667]])

    # aprox 36 meters from [691,163],[862,154] to [795, 667], [1026, 629] computed manually
    # and 8 meters from [691,163] to [862,154]
    TARGET_WIDTH: int = 8
    TARGET_HEIGHT: int = 36

    TARGET: np.ndarray = np.array(
        [
            [0, 0],
            [TARGET_WIDTH - 1, 0],
            [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
            [0, TARGET_HEIGHT - 1],
        ]
    )


DEFAULT_VID_SIZE_IN_MB = 50
