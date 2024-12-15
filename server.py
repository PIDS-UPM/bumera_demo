import cv2 as cv
import socket
import numpy as np
import datetime as dt
from modules.models import MoveNet, CNN_Bullying, Emotion_Detection
from modules.alarms import AlarmGenerator
from tensorflow import greater


def bullying_pipeline(frame, pose_model, prediction_model):
    cv.imshow("Received frames", frame)
    pred = pose_model(frame)
    cv.imshow("Prediction", pred)
    res = prediction_model.predict(prediction_model.preprocess(pred))["dense_1"]
    if greater(res, 0.5):
        counter += 1
        if counter >= 5:
            alert_gen.send_notifications(
                title="BULLYING WARNING", body="Possible bullying in the hallway"
            )
    else:
        counter = 0
    print(res)


def mood_pipeline(frame, full_model):
    cv.imshow("Received frames", frame)
    prediction = full_model(frame)
    cv.imshow("Prediction", prediction)


# Receiver configuration
hostname = socket.gethostname()
IP_REC = socket.gethostbyname(hostname)  # Receiver IP address (localhost: 127.0.0.1)
PORT_REC = 5001  # Receiver port where the messages will arrive
BUFFER_SIZE = 65535  # Max size of UDP messages

# Create the UDP socket
sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
sock.bind((IP_REC, PORT_REC))

# Alarms generator object
alert_gen = AlarmGenerator("key.json")
alert_gen.fetch_tokens("teachers", "fcmToken")

# Definition of the models to use
movenet = MoveNet("movenet", 224, 224)
cnn = CNN_Bullying("vgg19")
emotion_Detection = Emotion_Detection("emotions")


def main():
    print(
        f"Listening at {hostname}:{PORT_REC}. Waiting for frames... Press 'q' to exit."
    )
    while True:
        if DETECT_BULLYING:
            counter = 0
        data, _ = sock.recvfrom(BUFFER_SIZE)  # Receive UDP message
        frame = cv.imdecode(
            np.frombuffer(data, dtype=np.uint8), cv.IMREAD_COLOR
        )  # Decode received frame

        # Show the frame
        if frame is not None:
            (
                bullying_pipeline(frame, movenet, cnn)
                if DETECT_BULLYING
                else mood_pipeline(frame, emotion_Detection)
            )

        # Exit with 'q' key
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    sock.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # Choosing the objective of the code => True means bullying is going to be detected and False means mood is going to be detected
    DETECT_BULLYING = False
    main()
