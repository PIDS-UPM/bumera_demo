from mtcnn import MTCNN             # https://mtcnn.readthedocs.io/en/latest/
import tensorflow as tf
import tensorflow_hub as hub        # https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
from ultralytics import YOLO        # https://docs.ultralytics.com/es/tasks/pose/    ////    https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993
from abc import ABC, abstractmethod

print('Hola mundo :D')

#Creo que es MoveNet https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning-tflite-float16/1?tfhub-redirect=true
#Repo con Multi-MoveNet: https://github.com/ihalhashem/Multi-Person-Pose-Estimation

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
        pass
    
    @abstractmethod
    def predict(self, frame):
        pass

    @abstractmethod
    def postprocess(self, preds):
        pass

    def run_inference(self, frame):
        if not self.architecture:
            print('The class hasn\'t got an architecture defined! You must define one first.')
        processed_frame = self.preprocess(frame)
        predictions = self.predict(processed_frame)
        final_predictions = self.postprocess(predictions)
        return final_predictions


class MTCNN_face(Model):
    def __init__(self, name):
        super().__init__(name)
        self.architecture = self.load_model()

    def load_model(self):
        return MTCNN(stages='face_detection_only')     # The other option for stages is "face_and_landmarks_detection"
    
    def show_architecture(self):
        pass
    
    def preprocess(self, frame):
        return super().preprocess(frame)
    
    def predict(self, frame):
        return self.architecture.detect_faces(frame, box_format='xyxy')
    
    def postprocess(self, preds):
        return super().postprocess(preds)


class Mood_from_face(Model):
    ...


class YOLO_pose(Model):
    def __init__(self, name):
        super().__init__(name)
        self.architecture = YOLO('weights/yolo11n-pose.pt')

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

# pose = YOLO_pose('yolo')
# pose.load_model('weights/yolo11n-pose.pt')