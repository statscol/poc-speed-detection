import copy
import logging
import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from tqdm import tqdm

from config import CameraConfig, DetectorInferenceConfig
from utils import crop_image

inference_settings = DetectorInferenceConfig()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class BaseDetector(ABC):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def detect_image(self, img_path):
        """method to be used to process detection in images (raw image not video frames)"""
        pass

    @abstractmethod
    def process_videoframe(self, vid_path):
        """meant to be used as the default method to use when processing videoframes"""
        pass

    def get_outpath(self, image: Union[str, np.ndarray], action: str = "img"):
        """get output filepath from an image (array) or string to img file"""
        return os.path.join(
            self.out_folder,
            f"{self.model_name.split('/')[-1].split('.')[0]}_{action}_{uuid.uuid4()}.png"
            if not isinstance(image, str)
            else f"{self.model_name}_{action}_{Path(image).name}",
        )

    def crop_detection(self, image, box, save_crop=False, face=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if face else image
        crop = crop_image(box, image)
        if image is not None and save_crop:
            cv2.imwrite(self.get_outpath(image, action="crop"), crop)
        return crop

    def render_boxes(
        self,
        image,
        boxes,
        ids: Optional[np.ndarray],
        save_to_disk: bool = False,
        save_crop: bool = False,
        face=False,
        cls_names=None,
    ):
        """method to process cv2 boxes coming from a detect_image method in detector.py"""
        ids = ids if ids is not None else {i: "" for i in np.arange(0, len(boxes) - 1)}
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        img_out = copy.deepcopy(image)
        box_color = (0, 0, 255)
        for box, id in zip(boxes, ids):
            (xA, yA, xB, yB) = box
            crop = self.crop_detection(image, box, save_crop=save_crop, face=face)

            detection = (
                ids.get(id) if self.classifier is None else self.classifier.get(crop)[0]
            )
            # text_out=f"Id:{id} | Gender:{detection[0]} | Age: {detection[1]}" if face else f"Id:{id} Car body: {detection}"
            text_out = f"Id : {id} | Detection: {detection} "
            # draw bounding box
            cv2.rectangle(img_out, (xA, yA), (xB, yB), box_color, 2)
            # to account for negative values in the starting position when drawing text
            text_pos_w, text_pos_h = (
                xA,
                max(yA - inference_settings.OFFSET_BOX_CV2, 0),
            )
            text_size, _ = cv2.getTextSize(
                text_out, **inference_settings.CV2_DEFAULT_TEXT_BOX_ARGS
            )
            text_w, text_h = text_size
            cv2.rectangle(
                img_out,
                (text_pos_w, text_pos_h),
                (text_pos_w + text_w, text_pos_h + text_h),
                box_color,
                cv2.FILLED,
            )
            cv2.putText(
                img_out,
                text=text_out,
                # Align text with bg box source:
                # https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv
                org=(
                    text_pos_w,
                    text_pos_h
                    + text_h
                    + inference_settings.CV2_DEFAULT_TEXT_ARGS["fontScale"]
                    - 1,
                ),
                **inference_settings.CV2_DEFAULT_TEXT_ARGS,
            )

        if save_to_disk:
            cv2.imwrite(self.get_outpath(img_out), img_out)
        return img_out

    def get_video_writer(self, vid_path, cap_input, output_size: tuple = (840, 480)):
        fps = cap_input.get(cv2.CAP_PROP_FPS)
        self.logger.info(
            f"Saving video from source | FPS: {fps} | Resolution: {output_size}"
        )
        out_path = os.path.join(
            str(self.out_folder),
            f"{self.model_name.split('.')[0]}_{Path(vid_path).name}",
        )
        video_writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            output_size,
        )
        self.logger.info(f"Saving video to {out_path}")
        return video_writer, out_path

    def detect_video(
        self, vid_path: str, output_size: tuple = (840, 480), **kwargs
    ) -> None:
        """detects person in an mp4 video , saves output to a file resizing image to @param output_size"""

        cap = cv2.VideoCapture(vid_path)
        video_writer, out_path = self.get_video_writer(vid_path, cap, output_size)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Inference on video")
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                annotated_frame = self.process_videoframe(frame, **kwargs)
                frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_pos % inference_settings.FRAMES_UPDATE_ON_VIDEO == 0:
                    pbar.update(frame_pos)
                # resize to output_size
                annotated_frame = cv2.resize(
                    annotated_frame, output_size, interpolation=cv2.INTER_CUBIC
                )
                video_writer.write(annotated_frame)

            else:
                break

        cap.release()
        video_writer.release()
        self.logger.info(f"Saved video to {out_path}")

    def save_frames(self, vid_path: str, frame_skip: Optional[int] = 30):
        """Saves frames from a video, if @param frame_skip is None,
        all the frames are saved, or if its an integer,
        it will skip every @param frame_skip frames"""
        out_path = Path(
            os.path.join(self.out_folder), Path(vid_path).name.split(".")[0]
        )
        out_path.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(vid_path)
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                out_path_frame = str(out_path / f"frame{frame_pos}.png")
                cv2.imwrite(out_path_frame, frame)
                if frame_skip is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos + frame_skip)
                self.logger.info(f"Saved video to {out_path_frame}")
            else:
                break

        cap.release()

    def detect_camera(
        self, camera_config: Optional[CameraConfig], mode: str = "web", **kwargs
    ):
        # for laptops with single cameras
        assert mode in ["web", "rstp"], ValueError(
            "Mode must be either web or rstp protocol"
        )
        cap = cv2.VideoCapture(
            0
            if camera_config is None
            else f"rtsp://{camera_config.USERNAME}:{camera_config.PASSWORD}@{camera_config.URL}:554/stream1"
        )
        vid_dims = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        video_writer, out_path = self.get_video_writer(
            f"rstp_cam_{uuid.uuid4()}.mp4", cap, vid_dims
        )
        while True:
            # capture, process and display resulting image
            _, frame = cap.read()
            frame = self.process_videoframe(frame, **kwargs)
            video_writer.write(frame)
            cv2.imshow("frame", frame)

            # the 'ESC' key is set as the default key to exit the window
            if cv2.waitKey(20) & 0xFF == 27:
                break

        cap.release()
        video_writer.release()
        self.logger.info(f"Saved video to {out_path}")
        cv2.destroyAllWindows()
