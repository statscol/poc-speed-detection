import logging

import albumentations as ab
import cv2
import numpy as np
import onnxruntime


class FaceGenderage:
    def __init__(self, rec_name="./models/genderage_v1.onnx", outputs=None, **kwargs):
        self.rec_model = onnxruntime.InferenceSession(rec_name)
        self.input = self.rec_model.get_inputs()[0]
        self.id2gender = {0: "Female", 1: "Male"}
        if outputs is None:
            outputs = [e.name for e in self.rec_model.get_outputs()]
        self.outputs = outputs

    # warmup
    def prepare(self, **kwargs):
        logging.info("Warming up GenderAge ONNX Runtime engine...")
        self.rec_model.run(
            self.outputs,
            {
                self.rec_model.get_inputs()[0].name: [
                    np.zeros(tuple(self.input.shape[1:]), np.float32)
                ]
            },
        )

    def preprocess(img):
        # default size for this classifier (112,112) face crop
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
        # bgr to rbg
        img = img[:, :, ::-1].transpose(2, 0, 1)
        return img.astype(np.float32)

    def get(self, face_img):
        if not isinstance(face_img, list):
            face_img = [face_img]

        face_img = np.stack(face_img)
        imgs = face_img.copy()

        if face_img[0].shape != tuple(self.rec_model.get_inputs()[0].shape[1:]):
            input_size = self.rec_model.get_inputs()[0].shape[2:]
            imgs = cv2.dnn.blobFromImages(
                face_img, 1.0, input_size, (0.0, 0.0, 0.0), swapRB=True
            )

        _ga = []
        ret = self.rec_model.run(self.outputs, {self.input.name: imgs})[0]
        # Gender: 0-Female, 1-Male | Age, estimate from 0 to 100
        for e in ret:
            e = np.expand_dims(e, axis=0)
            g = e[:, 0:2].flatten()
            gender = np.argmax(g)
            a = e[:, 2:202].reshape((100, 2))
            a = np.argmax(a, axis=1)
            age = int(sum(a))
            _ga.append((self.id2gender[int(gender)], age))
        return _ga


class CarClassifier:
    def __init__(self, model_name: str = "./models/carDetect.onnx") -> None:
        self.model = onnxruntime.InferenceSession(model_name)
        self.input = self.model.get_inputs()[0]
        self.label2id = {
            "sedan": 0,
            "suv": 1,
            "van": 2,
            "truck": 3,
            "bus": 4,
            "hatchback": 5,
            "pick-up-truck": 6,
            "tricimoto": 7,
            "other": 8,
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.augmentations = ab.Compose(
            [
                ab.LongestMaxSize(
                    p=1, always_apply=True, max_size=224, interpolation=1
                ),
                ab.PadIfNeeded(
                    p=1,
                    always_apply=True,
                    min_height=224,
                    min_width=224,
                    border_mode=0,
                    value=0,
                ),
                ab.Normalize(
                    p=1,
                    always_apply=True,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.226, 0.226, 0.226),
                ),
            ]
        )
        self.outputs = [e.name for e in self.model.get_outputs()]

    def prepare(self, **kwargs):
        logging.info("Warming up GenderAge ONNX Runtime engine...")
        self.model.run(
            self.outputs,
            {
                self.model.get_inputs()[0].name: [
                    np.zeros(tuple(self.input.shape[1:]), np.float32)
                ]
            },
        )

    def preprocess(self, img):
        """Images in RGB Format:"""
        img = self.augmentations(image=img)
        img = img["image"].copy().transpose(2, 0, 1)
        return np.expand_dims(img, axis=0)

    def get(self, img):
        img = self.preprocess(img)
        pred = self.model.run(self.outputs, {self.input.name: img})[0]
        pred = np.argmax(pred, axis=1)[0].astype(int)
        return [self.id2label[pred]]
