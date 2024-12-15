import cv2 as cv
import socket
import numpy as np
from models import MoveNet, CNN_bullying, Emotion_detection
from alarms import AlarmGenerator


def bullying_pipeline(frame, pose_model, prediction_model, counter):
    cv.imshow("Received frames", frame)
    pred = pose_model(frame)
    cv.imshow("Prediction", pred)
    res = prediction_model.predict(prediction_model.preprocess(pred))
    res = next(iter(res.values()))
    res = res.numpy()[0, 0]
    if res > 0.5:
        counter[0] += 1
        if counter[0] >= 20:
            print("BULLYING !!!")
            alert_gen.send_notifications(
                title="BULLYING WARNING", body="Possible bullying in the hallway"
            )
    else:
        counter[0] = 0
    print(res, counter)


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
cnn = CNN_bullying("vgg19")
emotion_detection = Emotion_detection("emotions")


def main():
    print(
        f"Listening at {hostname}:{PORT_REC}. Waiting for frames... Press 'q' to exit."
    )
    if DETECT_BULLYING:
        counter = [0]
    while True:
        data, _ = sock.recvfrom(BUFFER_SIZE)  # Receive UDP message
        frame = cv.imdecode(
            np.frombuffer(data, dtype=np.uint8), cv.IMREAD_COLOR
        )  # Decode received frame

        # Show the frame
        if frame is not None:
            (
                bullying_pipeline(frame, movenet, cnn, counter)
                if DETECT_BULLYING
                else mood_pipeline(frame, emotion_detection)
            )

        # Exit with 'q' key
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    sock.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # Choosing the objective of the code => True means bullying is going to be detected and False means mood is going to be detected
    DETECT_BULLYING = True
    main()
