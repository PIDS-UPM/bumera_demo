# -----------------------------------------------------------------------------
# Author: Yago Boleas, Alberto Sánchez, Guillermo Pérez, Ana Mª Torres
# Project: Bumera
# Date: 17/12/2024
# Description: Python script for visualizing human poses from JSON files 
#              containing keypoint data. Keypoints are drawn as circles, and 
#              connections between keypoints are represented as lines to form 
#              a skeletal pose. The visualization is displayed using OpenCV on 
#              a blank image of predefined dimensions.
#
# License: This code is released under the MIT License.
#          You are free to use, modify, and distribute this software, provided
#          that proper credit is given to the original authors.
#
# Note: For more details, please refer to the LICENSE file included in the repository.
# -----------------------------------------------------------------------------

import json
import cv2 as cv
import numpy as np

# Constants
LABELS = (
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
IMG_HEIGHT = 512
IMG_WIDTH = 512

# Data collection
violence = False
num = 145  # 20, 57, 520, 740, 747, 836, 845, 860, 944, 957
file_name = f"poses_{num}.json"
with open(f'poses/{"" if violence else "Non"}Violence/{file_name}', "r") as file:
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
            cv.line(
                img,
                (int(pt1[1]), int(pt1[0])),
                (int(pt2[1]), int(pt2[0])),
                (0, 255, 0),
                3,
            )
    cv.imshow("Imagen negra", img)
    cv.waitKey(100)
cv.destroyAllWindows()
