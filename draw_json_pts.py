import json
import cv2 as cv
import numpy as np

# Constants
LABELS = ("nose", "left eye", "right eye", "left ear", "right ear", "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle",)
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
    ("right knee", "right ankle")
    ]
IMG_HEIGHT = 512
IMG_WIDTH = 512

# Data collection
violence = False
num = 520    # 20, 57, 520, 740, 747, 836, 845, 860, 944, 957
file_name = f'{"V" if violence else "NV"}_{num}.mp4'
with open(f'train_videos/{"" if violence else "Non"}Violence/{file_name}', 'rt') as file:
    data = json.load(file)

# Data representation
for frame in data:
    img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    for pose in frame.keys():
        keypoints = frame[pose]
        for body_part in keypoints.keys():
            point = keypoints[body_part]
            cv.circle(img, (int(point[1]), int(point[0])), 5, (0, 0, 255), 1)
        for connection in CONNECTIONS:
            pt1 = keypoints[connection[0]]
            pt2 = keypoints[connection[1]]
            cv.line(img, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])), (0, 255, 0), 5)
    cv.imshow('Imagen negra', img)
    cv.waitKey(100)
cv.destroyAllWindows()
