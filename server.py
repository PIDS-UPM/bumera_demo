import cv2 as cv
import socket
import numpy as np
from models import MoveNet, CNN_bullying, Emotion_detection


def bullying_pipeline(frame, pose_model, prediction_model):
    cv.imshow("Received frames", frame)
    pred = pose_model(frame)
    cv.imshow("Prediction", pred)
    res = prediction_model.predict(prediction_model.preprocess(pred))
    # print('Violence' if res >=0.5 else 'Non Violence')
    print(res["dense_1"])


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

# Definition of the models to use
movenet = MoveNet("movenet", 224, 224)
cnn = CNN_bullying("vgg19")
emotion_detection = Emotion_detection("emotions")

# Choosing the objective of the code
DETECT_BULLYING = False


def main():
    print(
        f"Listening at {hostname}:{PORT_REC}. Waiting for frames... Press 'q' to exit."
    )
    while True:

        data, _ = sock.recvfrom(BUFFER_SIZE)  # Receive UDP message
        frame = cv.imdecode(
            np.frombuffer(data, dtype=np.uint8), cv.IMREAD_COLOR
        )  # Decode received frame

        # Show the frame
        if frame is not None:
            (
                bullying_pipeline(frame, movenet, cnn)
                if DETECT_BULLYING
                else mood_pipeline(frame, emotion_detection)
            )

        # Exit with 'q' key
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    sock.close()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
