import cv2 as cv                    # https://docs.opencv.org/4.x/    ///    https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
import mtcnn                        # https://mtcnn.readthedocs.io/en/latest/
import tensorflow as tf
# import tensorflow_hub as hub
# from ultralytics import YOLO

def faces_in_frame(model, frame):
    faces = model.detect_faces(frame, box_format='xyxy')
    if len(faces) > 0:
        for face in faces:
            box = face['box']
            cv.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
            for key in face['keypoints'].keys():
                cv.circle(frame, (face['keypoints'][key][0], face['keypoints'][key][1]), 10, (255, 0, 0), 1)
    return frame

def pose_in_frame(model, frame): 
    raise NotImplementedError
    return result


# Open the default camera
cam = cv.VideoCapture(0)        # https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html
if not cam.isOpened():
    print("Cannot open camera")
    exit()


# Generate the models
face_detection = mtcnn.MTCNN()
# pose_detection = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')        # https://www.geeksforgeeks.org/human-pose-detection-using-movenet-with-tensorflowhub/
# yolo_model = YOLO('weights/yolo11n-pose.pt', 'pose')
pose_detecion = tf.saved_model.load('weights/movenet')

# Get the default frame width and height
frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))       # Máximo = 1080 // Usa 480 px
frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))         # Máximo = 1920 // Usa 640 px
print(f'Size of the window: {frame_width} × {frame_height} px')


# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('videos/output.mp4', fourcc, 20.0, (frame_width, frame_height))


# Choice of model and function
procesar_pixel = faces_in_frame
model = face_detection


# Loop for detection
while True:
    ret, frame = cam.read()     # bool, np.ndarray[uint8] (BGR mode)

    if not ret: 
        print('Can\'t receive frame (stream end?). Exiting ...')
        break

    # Our operations on the frame come here
    frame = procesar_pixel(model, frame)
    
    # Display the captured frame
    cv.imshow('Camera', frame)
    
    # Write the frame to the output file
    out.write(frame)

    # Press 'q' to exit the loop
    if cv.waitKey(1) == ord('q'):
        cam.release()
        out.release()
        cv.destroyAllWindows()
        break

# Release the capture and writer objects
_, im = cam.read()
cam.release()
out.release()
cv.destroyAllWindows()


# Using another frame for testing
cv.imshow('Last frame', im)
cv.waitKey(0)
cv.destroyAllWindows()

