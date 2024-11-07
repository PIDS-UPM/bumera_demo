import cv2 as cv                    # https://docs.opencv.org/4.x/    ///    https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np

CONNECTIONS = [
    ('nose', 'left eye'), ('left eye', 'left ear'), ('nose', 'right eye'), ('right eye', 'right ear'), ('nose', 'left shoulder'), ('left shoulder', 'left elbow'), 
    ('left elbow', 'left wrist'), ('nose', 'right shoulder'), ('right shoulder', 'right elbow'), ('right elbow', 'right wrist'), ('left shoulder', 'left hip'), 
    ('right shoulder', 'right hip'), ('left hip', 'right hip'), ('left hip', 'left knee'), ('right hip', 'right knee')
    ]

def show_pred(img, keypoints_dict, label, save=False):
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img[0])
    for key in keypoints_dict.keys():
        keypoints = keypoints_dict[key]
        plt.subplot(1, 3, 2)
        plt.imshow(img[0])
        plt.title('Pose')
        plt.axis('off')
        for i in range(13):
            plt.scatter(keypoints[label[i]][1], keypoints[label[i]][0], color='green')

        for start_key, end_key in CONNECTIONS:
            if start_key in keypoints and end_key in keypoints:
                start_point = keypoints[start_key][:2]  # Take first two values
                end_point = keypoints[end_key][:2]      # Take first two values
                plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], linewidth=2)
        
        plt.subplot(1, 3, 3)
        plt.imshow((img[0]/255)/255)
        plt.title('Only Pose Image')
        for start_key, end_key in CONNECTIONS:
            if start_key in keypoints and end_key in keypoints:
                start_point = keypoints[start_key][:2]  # Take first two values
                end_point = keypoints[end_key][:2]      # Take first two values
                plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], linewidth=2)
    if save: plt.savefig(f'images/last_infer.jpg')
    return fig

def detect_keypoints(model, frame):
    # Preprocessing the frame
    X = tf.expand_dims(frame, axis=0)
    X = tf.cast(tf.image.resize_with_pad(X, 256, 256), dtype=tf.int32)

    # People detection
    pred = model(X)['output_0'].numpy()
    people_detected = np.where(pred[0, :, 55] > 0.2)    # Extraemos los índices donde detectamos muy posiblemente a gente

    # Pose detection and saving
    keypoints_dict = dict()
    for key in people_detected[0]:
        points = (pred[0, key, :]*256).astype(float)
        keypoints = dict()
        for i in range(0, len(points)-5, 3):
            keypoints[label[i//3]] = [p for p in points[i:i+3]]
        keypoints_dict[key] = keypoints
    
    return keypoints_dict

def draw_pts_opencv(keypoints_dict, frame):
    for key in keypoints_dict.keys():
        keypoints = keypoints_dict[key]
        for body_part in keypoints.keys():
            point = keypoints[body_part]
            cv.circle(frame, (int(point[1]/256*640), int(point[0]/256*480)), 5, (0, 0, 255), 1)
        for connection in CONNECTIONS:
            pt1 = keypoints[connection[0]]
            pt2 = keypoints[connection[1]]
            cv.line(frame, (int(pt1[1]/256*640), int(pt1[0]/256*480)), (int(pt2[1]/256*640), int(pt2[0]/256*480)), (0, 255, 0), 2)


    return frame

# SIGUIENDO EL TUTORIAL DE https://www.kaggle.com/code/mohitgupta12/pose-detection-using-mobilenet
# ESTE PUEDE SERVIR https://www.kaggle.com/code/ibrahimserouis99/human-pose-estimation-with-movenet/notebook
movenet = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/multipose-lightning/1").signatures['serving_default']

label = ["nose", "left eye", "right eye", "left ear", "right ear", "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]

cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Cannot open camera")
    exit()

# name = 'agresion'
# im = plt.imread(f'images/{name}.jpg')
# plt.imshow(im)

while True:
    _, im = cam.read()
# cam.release()
# cv.imshow('Photo', im)
# cv.waitKey(0)
# X = tf.expand_dims(im, axis=0)
# X = tf.cast(tf.image.resize_with_pad(X, 256, 256), dtype=tf.int32)

# pred = movenet(X)['output_0'].numpy()
# people_detected = np.where(pred[0, :, 55] > 0.2)    # Extraemos los índices donde detectamos muy posiblemente a gente
# max_key = pred[0, :, 55].argmax()

# keypoints_dict = dict()
# for key in people_detected[0]:
#     points = (pred[0, key, :]*256).astype(float)
#     keypoints = dict()
#     for i in range(0, len(points)-5, 3):
#         keypoints[label[i//3]] = [p for p in points[i:i+3]]
#     keypoints_dict[key] = keypoints
    keypoints_dict = detect_keypoints(movenet, im)
    result = draw_pts_opencv(keypoints_dict, im)
    cv.imshow('Pose', im)
    if cv.waitKey(1) == ord('q'):
        cam.release()
        cv.destroyAllWindows()
        break

# img = tf.image.resize_with_pad(im, 256, 256)
# img = tf.cast(img, dtype=tf.int32)
# img = tf.expand_dims(img, axis=0)
# img = img.numpy()
# summary = show_pred(img, keypoints_dict, label)
# plt.show()

# cv.imshow('MoveNet', im)
# cam.release()
# cv.waitKey(0)
# cv.destroyAllWindows()

