# -----------------------------------------------------------------------------
# Author: Yago Boleas, Alberto Sánchez, Guillermo Pérez, Ana Mª Torres
# Project: Bumera
# Date: 17/12/2024
# Description: Description: Python module implementing various machine learning
#              models for face detection (MTCNN), emotion recognition (FER),
#              pose estimation (MoveNet), and CNN-based bullying detection.
#              Includes an abstract base class for standardizing model
#              interaction and preprocessing pipelines.
#
# License: This code is released under the MIT License.
#          You are free to use, modify, and distribute this software, provided
#          that proper credit is given to the original authors.
#
# Note: For more details, please refer to the LICENSE file included in the repository.
# -----------------------------------------------------------------------------

from mtcnn import MTCNN  # https://mtcnn.readthedocs.io/en/latest/
from fer import FER
import tensorflow as tf
import tensorflow_hub as hub  # https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
from abc import ABC, abstractmethod
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw


class Model(ABC):
    def __init__(self, name: str):
        """
        Initializes the Model object.

        :param name: str : The name of the model.
        """
        self.name = name
        self.architecture = None

    @abstractmethod
    def load_model(self):
        """
        Abstract method to load the model.
        """
        pass

    @abstractmethod
    def preprocess(self, frame):
        """
        Abstract method to preprocess the input frame.

        :param frame: The input frame to be preprocessed.
        :return: The preprocessed frame.
        """
        return frame

    @abstractmethod
    def predict(self, frame):
        """
        Abstract method to make predictions on the input frame.

        :param frame: The input frame to make predictions on.
        """
        pass

    @abstractmethod
    def postprocess(self, frame, preds):
        """
        Abstract method to postprocess the predictions.

        :param frame: The input frame.
        :param preds: The predictions made by the model.
        :return: The final predictions after postprocessing.
        """
        return frame

    def __call__(self, frame):
        """
        Call method to process the frame through the model pipeline.

        :param frame: The input frame to be processed.
        :return: The final predictions after processing.
        """
        processed_frame = self.preprocess(frame)
        predictions = self.predict(processed_frame)
        final_predictions = self.postprocess(frame, predictions)
        return final_predictions


class MTCNN_Face(Model):
    def __init__(self, name: str, only_face: bool = True):
        """
        Initializes the MTCNN_Face object.

        :param name: str : The name of the model.
        :param only_face: bool : Flag to detect only faces or both faces and landmarks.
        """
        super().__init__(name)
        self.only_face = only_face
        self.architecture = self.load_model()

    def load_model(self):
        """
        Loads the MTCNN model.

        :return: The loaded MTCNN model.
        """
        return MTCNN(
            stages=(
                "face_detection_only"
                if self.only_face
                else "face_and_landmarks_detection"
            )
        )

    def preprocess(self, frame):
        """
        Preprocesses the input frame.

        :param frame: The input frame to be preprocessed.
        :return: The preprocessed frame.
        """
        return super().preprocess(frame)

    def predict(self, frame):
        """
        Makes predictions on the input frame using the MTCNN model.

        :param frame: The input frame to make predictions on.
        :return: The predictions made by the MTCNN model.
        """
        prediction = self.architecture.detect_faces(frame, box_format="xyxy")
        return prediction

    def postprocess(self, frame, preds):
        """
        Postprocesses the predictions by drawing bounding boxes and keypoints on the frame.

        :param frame: The input frame.
        :param preds: The predictions made by the model.
        :return: The final frame with drawn bounding boxes and keypoints.
        """
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


class Emotion_Detection(Model):
    def __init__(self, name: str, show_roi: bool = True):
        """
        Initializes the Emotion_Detection object.

        :param name: str : The name of the model.
        :param show_roi: bool : Flag to show the region of interest.
        """
        self.name = name
        self.face_detector = MTCNN_Face("mtcnn")
        self.emotion_detector = FER(mtcnn=False)
        self.show_roi = show_roi
        self.box_expansion = 0.2
        self.colors = {
            "angry": (0, 0, 255),
            "disgust": (0, 140, 255),
            "fear": (0, 255, 255),
            "happy": (0, 255, 0),
            "sad": (255, 0, 0),
            "surprise": (255, 0, 255),
            "neutral": (255, 255, 255),
        }  # BGR colors for each emotion

    def _expand_roi(self, frame, face):
        """
        Extracts and expands the face region of interest for the FER model.

        :param frame: The input frame.
        :param face: The detected face.
        :return: The expanded region of interest.
        """
        x1, y1, x2, y2 = face["box"]
        frame_h, frame_w = frame.shape[:2]
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        x1 = max(0, int(x1 - self.box_expansion * x1))
        y1 = max(0, int(y1 - self.box_expansion * y1))
        x2 = min(int(x2 + self.box_expansion * x2), frame_w)
        y2 = min(int(y2 + self.box_expansion * y2), frame_h)
        return [x1, y1, x2, y2]

    def _pick_face(self, frame, box):
        """
        Picks the face region from the frame.

        :param frame: The input frame.
        :param box: The bounding box of the face.
        :return: The face region.
        """
        x1, y1, x2, y2 = box
        return frame[y1:y2, x1:x2]

    def _draw_info(self, frame, box, index, feeling):
        """
        Draws the emotion information on the frame.

        :param frame: The input frame.
        :param box: The bounding box of the face.
        :param index: The index of the face.
        :param feeling: The detected emotion.
        :return: The frame with drawn emotion information.
        """
        frame = cv.rectangle(
            img=frame,
            pt1=(box[0], box[1]),
            pt2=(box[2], box[3]),
            color=self.colors[feeling],
        )
        frame = cv.putText(
            img=frame,
            text=f"Face {index}: {feeling}",
            org=(box[0], box[1]),
            fontFace=0,
            fontScale=0.5,
            color=self.colors[feeling],
        )
        return frame

    def load_model(self):
        """
        Loads the emotion detection model.

        :return: The loaded emotion detection model.
        """
        return super().load_model()

    def preprocess(self, frame):
        """
        Preprocesses the input frame using the face detector.

        :param frame: The input frame to be preprocessed.
        :return: The preprocessed frame.
        """
        return self.face_detector.preprocess(frame)

    def predict(self, frame):
        """
        Makes predictions on the input frame using the face detector.

        :param frame: The input frame to make predictions on.
        :return: The predictions made by the face detector.
        """
        faces = self.face_detector.predict(frame)
        return faces

    def postprocess(self, frame, preds):
        """
        Postprocesses the predictions by detecting emotions and drawing information on the frame.

        :param frame: The input frame.
        :param preds: The predictions made by the model.
        :return: The final frame with drawn emotion information.
        """
        if len(preds) < 0:
            return frame
        for i, pred in enumerate(preds):
            roi = self._expand_roi(frame, pred)
            face = self._pick_face(frame, roi)
            feelings = self.emotion_detector.detect_emotions(face, [roi])
            top_feeling = [
                max(x["emotions"], key=lambda key: x["emotions"][key]) for x in feelings
            ]
            if len(top_feeling):
                frame = self._draw_info(frame, roi, i, top_feeling[0])

        return frame


class MoveNet(Model):
    def __init__(
        self,
        name: str,
        img_width: int = 640,
        img_height: int = 480,
        pred_confidence: float = 0.2,
        pt_confidence: float = 0.2,
    ):
        """
        Initializes the MoveNet object.

        :param name: str : The name of the model.
        :param img_width: int : The width of the input image.
        :param img_height: int : The height of the input image.
        :param pred_confidence: float : The prediction confidence threshold.
        :param pt_confidence: float : The point confidence threshold.
        """
        super().__init__(name)
        self.architecture = self.load_model()
        self.img_width = img_width
        self.img_height = img_height
        self.pred_confidence = pred_confidence
        self.pt_confidence = pt_confidence
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
        """
        Loads the MoveNet model.

        :return: The loaded MoveNet model.
        """
        return hub.load(
            "https://tfhub.dev/google/movenet/multipose/lightning/1"
        ).signatures["serving_default"]

    def preprocess(self, frame):
        """
        Preprocesses the input frame.

        :param frame: The input frame to be preprocessed.
        :return: The preprocessed frame.
        """
        X = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        X = tf.expand_dims(frame, axis=0)
        X = tf.cast(tf.image.resize_with_pad(X, 256, 256), dtype=tf.int32)
        return X

    def predict(self, frame):
        """
        Makes predictions on the input frame using the MoveNet model.

        :param frame: The input frame to make predictions on.
        :return: The predictions made by the MoveNet model.
        """
        prediction = self.architecture(frame)
        prediction = prediction["output_0"].numpy()
        people_detected = np.where(prediction[0, :, 55] >= self.pred_confidence)
        keypoints_dict = dict()
        for key in people_detected[0]:
            points = (prediction[0, key, :] * 256).astype(float)
            keypoints = dict()
            for i in range(0, len(points) - 5, 3):
                keypoints[self.labels[i // 3]] = [p for p in points[i : i + 3]]
                keypoints[self.labels[i // 3]][2] = (
                    keypoints[self.labels[i // 3]][2] * 100 / 256
                )
            keypoints_dict[int(key)] = keypoints
        return keypoints_dict

    def postprocess(self, frame, preds, to_cnn: bool = True):
        """
        Postprocesses the predictions by drawing keypoints on the frame.

        :param frame: The input frame.
        :param preds: The predictions made by the model.
        :param to_cnn: bool : Flag to reshape the output to CNN format.
        :return: The final frame with drawn keypoints.
        """
        img = Image.fromarray(
            np.zeros((224, 224, 3) if to_cnn else frame, dtype=np.uint8)
        )
        draw = ImageDraw.Draw(img)
        new_frame = self.draw_points(draw, preds)
        return np.reshape(np.asarray(new_frame.im, dtype=np.uint8), (224, 224, 3))

    def draw_points(self, frame, keypoints_dict):
        """
        Draws keypoints on the frame.

        :param frame: The input frame.
        :param keypoints_dict: The dictionary of keypoints.
        :return: The frame with drawn keypoints.
        """
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
            body_parts = list()
            for body_part in keypoints.keys():
                point = keypoints[body_part]
                if point[2] >= self.pt_confidence:
                    body_parts.append(body_part)
                    frame.ellipse(
                        (
                            int(point[1]) - 3,
                            int(point[0]) - 3,
                            int(point[1]) + 3,
                            int(point[0]) + 3,
                        ),
                        outline="red",
                        width=1,
                    )
            my_connections = [
                con
                for con in CONNECTIONS
                if con[0] in body_parts and con[1] in body_parts
            ]
            for connection in my_connections:
                pt1 = keypoints[connection[0]]
                pt2 = keypoints[connection[1]]
                frame.line(
                    (int(pt1[1]), int(pt1[0]), int(pt2[1]), int(pt2[0])),
                    fill=[
                        "orange",
                        "yellow",
                        "lime",
                        "aqua",
                        "blue",
                        "magenta",
                    ][int(int(key) % 6)],
                    width=3,
                )
        return frame


class CNN_Bullying(Model):
    def __init__(self, name: str):
        """
        Initializes the CNN_Bullying object.

        :param name: str : The name of the model.
        """
        super().__init__(name)
        self.architecture = self.load_model()

    def load_model(self):
        """
        Loads the CNN bullying detection model.

        :return: The loaded CNN bullying detection model.
        """
        return tf.saved_model.load(f"models/{self.name.lower()}").signatures[
            "serving_default"
        ]

    def preprocess(self, frame):
        """
        Preprocesses the input frame.

        :param frame: The input frame to be preprocessed.
        :return: The preprocessed frame.
        """
        match self.name:
            case "vgg16":
                from tensorflow.keras.applications.vgg16 import preprocess_input
            case "vgg19":
                from tensorflow.keras.applications.vgg19 import preprocess_input
            case "efficientnet_v2":
                from tensorflow.keras.applications.efficientnet_v2 import (
                    preprocess_input,
                )
            case "inception_resnet_v2":
                from tensorflow.keras.applications.inception_resnet_v2 import (
                    preprocess_input,
                )
            case "convnext":
                from tensorflow.keras.applications.convnext import preprocess_input
        return preprocess_input(frame)

    def predict(self, frame):
        """
        Makes predictions on the input frame using the CNN model.

        :param frame: The input frame to make predictions on.
        :return: The predictions made by the CNN model.
        """
        return self.architecture(tf.expand_dims(frame, axis=0))

    def postprocess(self, frame, preds):
        """
        Postprocesses the predictions.

        :param frame: The input frame.
        :param preds: The predictions made by the model.
        :return: The final predictions after postprocessing.
        """
        return super().postprocess(frame, preds)
