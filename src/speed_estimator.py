import logging
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from classifier import CarClassifier
from config import SpeedEstimationConfig
from utils import ViewTransformer, crop_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


speed_config = SpeedEstimationConfig()


class SpeedEstimator:
    """Class to estimate a car speed in a Region of Interest (ROI)
    in a video input,
    - most of the methods were taken from a
    Roboflow tutorial https://blog.roboflow.com/estimate-speed-computer-vision/
    """

    def __init__(
        self,
        vid_path: str,
        roi: np.array,
        target: np.array,
        model_name: str = "yolov8s.pt",
        detector_classes: List[int] = [2],
        classifier_path: str = "./models/carDetect.onnx",
    ):
        self.vid_path = vid_path
        self.ROI = roi
        self.TARGET = target
        # min distance to trigger speed calculation
        self.THRESH_MIN_DISTANCE = 1.5
        self.polygon_transformer = ViewTransformer(self.ROI, self.TARGET)
        # default model
        self.model = YOLO(model_name)
        self.detector_classes = detector_classes
        self.classifier_path = classifier_path
        self.classifier = (
            CarClassifier(self.classifier_path)
            if self.classifier_path is not None
            else self.classifier_path
        )
        # reading input video
        self.cap = cv2.VideoCapture(self.vid_path)
        assert self.cap.isOpened(), "Error reading video file"
        self.w, self.h, self.fps = (
            int(self.cap.get(x))
            for x in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            )
        )

        # bounding box,track,label formatting
        self.thickness = sv.calculate_dynamic_line_thickness(
            resolution_wh=(self.w, self.h)
        )
        self.text_scale = sv.calculate_dynamic_text_scale(
            resolution_wh=(self.w, self.h)
        )
        # Bounding Box, Track, Label annotators
        self.bounding_box_annotator = sv.RoundBoxAnnotator(
            thickness=self.thickness, color_lookup=sv.ColorLookup.TRACK
        )

        self.label_annotator = sv.LabelAnnotator(
            text_scale=self.text_scale,
            text_thickness=self.thickness,
            text_position=sv.Position.BOTTOM_CENTER,
            color_lookup=sv.ColorLookup.TRACK,
        )
        self.track_annotator = sv.TraceAnnotator(
            thickness=self.thickness,
            trace_length=int(self.fps),
            position=sv.Position.BOTTOM_CENTER,
            color_lookup=sv.ColorLookup.TRACK,
        )

        self.polygon_zone = sv.PolygonZone(
            polygon=self.ROI, frame_resolution_wh=(self.w, self.h)
        )

        # Accumulates every car track using a dictionary with a deque of len equal to the number
        # of frames per second
        self.car_coordinates = defaultdict(lambda: deque(maxlen=int(self.fps)))

    def classify_detections(self, detections, frame):
        boxes, tracks, names = (
            detections.xyxy,
            detections.tracker_id,
            detections.class_id,
        )
        # account for frames where there are no detections
        if boxes is None or self.classifier_path is None:
            return names

        preds = []
        for box, _ in zip(boxes, tracks):
            crop = crop_image(box, frame)
            preds.append(self.classifier.get(crop)[0])
        return preds

    def estimate_speed(
        self,
        out_path: str,
        conf: float,
        iou_thresh: float,
        draw_roi: bool = False,
        show_output: bool = True,
        limit_seconds: Optional[int] = 60,
    ):
        """Estimates speed of vehicles in a video based on a predefined ROI

        Args:
            out_path (str): folder to save inference results
            conf (float): confidence for Yolo model
            iou_thresh (float): IOU threshold for Yolo model
            draw_roi (bool, optional): Whether or not to draw the ROI in the frames. Defaults to False.
            show_output (bool, optional): Whether or not to show the output on screen. Defaults to True.
            limit_seconds (Optional[int], optional): Number of seconds to use from the input video. Defaults to 60.
        """

        # make sure the out_path exists
        Path(out_path).mkdir(exist_ok=True, parents=True)

        # Tracking algorithm
        byte_track = sv.ByteTrack(frame_rate=self.fps, track_thresh=conf)
        # Video writer
        out_filepath = str(
            Path(out_path)
            / f"speed_{datetime.now().strftime('%H-%M-%S')}_{Path(self.vid_path).name}"
        )
        vid_size = self.w, self.h
        video_writer = cv2.VideoWriter(
            out_filepath, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, vid_size
        )

        while self.cap.isOpened():
            success, im0 = self.cap.read()
            if not success:
                break

            detections = self.model(
                im0,
                show=False,
                conf=conf,
                iou=iou_thresh,
                verbose=False,
                agnostic_nms=True,
                classes=self.detector_classes,
            )[0]

            img_raw = im0.copy()
            if draw_roi:
                im0 = sv.draw_polygon(
                    scene=im0.copy(), polygon=self.ROI, color=sv.Color.RED, thickness=4
                )

            detections = sv.Detections.from_ultralytics(detections)
            # filter out detections by class and confidence
            detections = detections[detections.confidence > conf]
            # filter out detections outside the zone
            detections = detections[self.polygon_zone.trigger(detections)]
            # run NMS again (this can be skipped)
            detections = detections.with_nms(iou_thresh)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = self.polygon_transformer.transform_points(points).astype(int)
            annotated_frame = im0.copy()

            classifier_labels = self.classify_detections(detections, img_raw)
            if len(detections) > 0:
                for tracker_id, cls_id, [x, y] in zip(
                    detections.tracker_id, classifier_labels, points
                ):
                    self.car_coordinates[tracker_id].append((x, y))

                    # format labels
                    labels = []

                    for tracker_id in detections.tracker_id:
                        # avoid flickering issues, only compute speed if we have half the number of
                        # observations compared to the frames per second (30 fps ->15)
                        # if the observation is a float means that the classifier was not set
                        cls_name = (
                            self.model.names.get(int(cls_id))
                            if isinstance(cls_id, float)
                            else cls_id
                        )
                        if len(self.car_coordinates[tracker_id]) < int(self.fps / 2):
                            labels.append(f"{cls_name} #{tracker_id}")
                        else:
                            # calculate speed
                            _, coordinate_start_y = self.car_coordinates[tracker_id][-1]
                            _, coordinate_end_y = self.car_coordinates[tracker_id][0]
                            # this accounts for two-way roads
                            distance_y = abs(coordinate_start_y - coordinate_end_y)
                            time = len(self.car_coordinates[tracker_id]) / self.fps
                            # to compute speed we use the displacement in the y axis
                            if distance_y < self.THRESH_MIN_DISTANCE:
                                distance_y = 0
                            # PersTransform returns difference in meters, and the time unit is seconds
                            # to convert to Km/h we multiply for 3600 s/1000m =3.6
                            speed = distance_y / time * 3.6
                            speed_txt_out = (
                                f"{speed:.1f} Km/h" if speed > 0 else "Stopped"
                            )
                            labels.append(f"{cls_id} #{tracker_id} : {speed_txt_out} ")

                annotated_frame = self.track_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )
                annotated_frame = self.bounding_box_annotator.annotate(
                    scene=annotated_frame, detections=detections
                )

                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame, detections=detections, labels=labels
                )
            video_writer.write(annotated_frame)
            # cv2.imwrite(
            #     f"./inference/frame_{int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))}.png",
            #     annotated_frame,
            # )
            if show_output:
                cv2.imshow("speed-app-img", annotated_frame)

            if (limit_seconds is not None) and (
                self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps
            ) > limit_seconds:
                # record only the first minute
                logging.info("Number of seconds reached...")
                break
            # logging.info(f"FRAME {self.cap.get(cv2.CAP_PROP_POS_FRAMES)}")
            # the 'ESC' key is set as the default key to exit the window
            if cv2.waitKey(20) & 0xFF == 27:
                break

        video_writer.release()
        self.cap.release()
        cv2.destroyAllWindows()
        logging.info(f"Saved video to {out_filepath}")


if __name__ == "__main__":
    estimation_config = SpeedEstimationConfig()
    detector = SpeedEstimator(
        "./data/video_test.mp4",
        estimation_config.ROI,
        estimation_config.TARGET,
        model_name=estimation_config.DETECTOR_PATH,
        classifier_path=estimation_config.CLASSIFIER_PATH,
        detector_classes=estimation_config.YOLO_DEFAULT_CLASSES,
    )
    detector.estimate_speed(
        "./inference",
        estimation_config.DEFAULT_CONF_SCORE,
        estimation_config.DEFAULT_IOU_THRESHOLD,
        draw_roi=estimation_config.DRAW_ROI,
        show_output=False,
        limit_seconds=60,
    )
