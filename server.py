import cv2 as cv
import socket
import numpy as np
from models import MoveNet, CNN_bullying

# Receiver configuration
hostname = socket.gethostname()
IP_REC = socket.gethostbyname(hostname)  # Receiver IP address (localhost: 127.0.0.1)
PORT_REC = 5001  # Receiver port where the messages will arrive
BUFFER_SIZE = 65535  # Max size of UDP messages

# Create the UDP socket
sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
sock.bind((IP_REC, PORT_REC))

movenet = MoveNet("movenet", 224, 224)
cnn = CNN_bullying('vgg19')
print(f"Listening at {hostname}:{PORT_REC}. Waiting for frames... Press 'q' to exit.")
while True:

    data, _ = sock.recvfrom(BUFFER_SIZE)  # Receive UDP message
    frame = cv.imdecode(
        np.frombuffer(data, dtype=np.uint8), cv.IMREAD_COLOR
    )  # Decode received frame

    # Show the frame
    if frame is not None:
        cv.imshow("Received frames", frame)
        pred = movenet(frame)
        cv.imshow("Prediction", pred)
        res = cnn.predict(cnn.preprocess(pred))
        # print('Violence' if res >=0.5 else 'Non Violence') 
        print(res['dense_1'])

    # Exit with 'q' key
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

sock.close()
cv.destroyAllWindows()
