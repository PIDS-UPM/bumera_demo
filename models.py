from mtcnn import MTCNN  # https://mtcnn.readthedocs.io/en/latest/
import tensorflow as tf
import tensorflow_hub as hub  # https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
from ultralytics import (
    YOLO,
)  # https://docs.ultralytics.com/es/tasks/pose/    ////    https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993
from abc import ABC, abstractmethod
import numpy as np
import cv2 as cv

# Creo que es MoveNet https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning-tflite-float16/1?tfhub-redirect=true
# Repo con Multi-MoveNet: https://github.com/ihalhashem/Multi-Person-Pose-Estimation


class Model(ABC):
    def __init__(self, name):
        self.name = name
        self.architecture = None
        self.hyperparameters = None

    @abstractmethod
    def show_architecture(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def preprocess(self, frame):
        return frame

    @abstractmethod
    def predict(self, frame):
        pass

    @abstractmethod
    def postprocess(self, frame, preds):
        return frame

    def __call__(self, frame):
        if not self.architecture:
            print(
                "The class hasn't got an architecture defined! You must define one first."
            )
        processed_frame = self.preprocess(frame)
        predictions = self.predict(processed_frame)
        final_predictions = self.postprocess(frame, predictions)
        return final_predictions


class MTCNN_Face(Model):
    def __init__(self, name, only_face=True):
        super().__init__(name)
        self.only_face = only_face
        self.architecture = self.load_model()

    def load_model(self):
        return MTCNN(
            stages=(
                "face_detection_only"
                if self.only_face
                else "face_and_landmarks_detection"
            )
        )  # The other option for stages is 'face_and_landmarks_detection'

    def show_architecture(self):
        pass

    def preprocess(self, frame):
        return super().preprocess(frame)

    def predict(self, frame):
        return self.architecture.detect_faces(frame, box_format="xyxy")

    def postprocess(self, frame, preds):
        if len(preds) > 0:
            for face in preds:
                if face["confidence"] >= 0.9:
                    box = face["box"]
                    cv.rectangle(
                        frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1
                    )
                    if "keypoints" in face.keys():
                        for key in face["keypoints"].keys():
                            cv.circle(
                                frame,
                                (face["keypoints"][key][0], face["keypoints"][key][1]),
                                10,
                                (255, 0, 0),
                                1,
                            )
        return frame


class Mood_from_face(Model): ...


class YOLO_pose(Model):
    def __init__(self, name):
        super().__init__(name)
        self.architecture = YOLO("weights/yolo11n-pose.pt")

    def load_model(self, path):
        return YOLO(path)

    def show_architecture(self):
        print(self.architecture.model)

    def preprocess(self, frame):
        return super().preprocess(frame)

    def predict(self, frame):
        return self.architecture(frame)

    def postprocess(self, preds):
        return super().postprocess(preds)

    def draw_points(self):
        raise NotImplementedError


class MoveNet(Model):
    def __init__(self, name):
        super().__init__(name)
        self.architecture = self.load_model()
        self.labels = (
            "nose",
            "left eye",
            "right eye",
            "left ear",
            "right ear",
            "left shoulder",
            "right shoulder",
            "left elbow",
            "right elbow",
            "left wrist",
            "right wrist",
            "left hip",
            "right hip",
            "left knee",
            "right knee",
            "left ankle",
            "right ankle",
        )

    def load_model(self):
        return hub.load(
            "https://tfhub.dev/google/movenet/multipose/lightning/1"
        ).signatures["serving_default"]

    def show_architecture(self):
        return super().show_architecture()

    def preprocess(self, frame):
        X = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        X = tf.expand_dims(frame, axis=0)
        X = tf.cast(tf.image.resize_with_pad(X, 256, 256), dtype=tf.int32)
        return X

    def predict(self, frame, threshold=0.2):
        prediction = self.architecture(frame)
        prediction = prediction["output_0"].numpy()
        people_detected = np.where(prediction[0, :, 55] > threshold)
        keypoints_dict = dict()
        for key in people_detected[0]:
            points = (prediction[0, key, :] * 256).astype(float)
            keypoints = dict()
            for i in range(0, len(points) - 5, 3):
                keypoints[self.labels[i // 3]] = [p for p in points[i : i + 3]]
            keypoints_dict[key] = keypoints
        return keypoints_dict

    def postprocess(self, frame, preds):
        # Aquí aún hay que detectar el bullying <--
        new_frame = self.draw_points(frame, preds)
        return new_frame

    def draw_points(self, frame, keypoints_dict):
        CONNECTIONS = [
            ("nose", "left eye"),
            ("left eye", "left ear"),
            ("nose", "right eye"),
            ("right eye", "right ear"),
            ("nose", "left shoulder"),
            ("left shoulder", "left elbow"),
            ("left elbow", "left wrist"),
            ("nose", "right shoulder"),
            ("right shoulder", "right elbow"),
            ("right elbow", "right wrist"),
            ("left shoulder", "left hip"),
            ("right shoulder", "right hip"),
            ("left hip", "right hip"),
            ("left hip", "left knee"),
            ("right hip", "right knee"),
            ("left knee", "left ankle"),
            ("right knee", "right ankle"),
        ]
        for key in keypoints_dict.keys():
            keypoints = keypoints_dict[key]
            for body_part in keypoints.keys():
                point = keypoints[body_part]
                cv.circle(
                    frame,
                    (int(point[1] / 256 * 640), int(point[0] / 256 * 480)),
                    5,
                    (0, 0, 255),
                    1,
                )
            for connection in CONNECTIONS:
                pt1 = keypoints[connection[0]]
                pt2 = keypoints[connection[1]]
                cv.line(
                    frame,
                    (int(pt1[1] / 256 * 640), int(pt1[0] / 256 * 480)),
                    (int(pt2[1] / 256 * 640), int(pt2[0] / 256 * 480)),
                    (0, 255, 0),
                    2,
                )
        return frame

    # def __call__(self, frame):
    #     if not self.architecture:
    #         print(
    #             "The class hasn't got an architecture defined! You must define one first."
    #         )
    #     processed_frame = self.preprocess(frame)
    #     predictions = self.predict(processed_frame)
    #     drawn_frame = self.postprocess(frame, predictions)
    #     return drawn_frame
