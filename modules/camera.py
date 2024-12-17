# -----------------------------------------------------------------------------
# Author: Yago Boleas, Alberto Sánchez, Guillermo Pérez, Ana Mª Torres
# Project: Bumera
# Date: 17/12/2024
# Description: This script defines a Camera class that manages video capture using OpenCV.
#              It supports real-time streaming of frames over a UDP socket to a specified IP
#              and port, saving video recordings to a file, and monitoring the camera's status.
#              The class includes methods to start/stop filming, reload the camera,
#              check FPS, and handle frame encoding for efficient transmission.
#              Designed for video-based applications, such as remote monitoring systems.
#
# License: This code is released under the MIT License.
#          You are free to use, modify, and distribute this software, provided
#          that proper credit is given to the original authors.
#
# Note: For more details, please refer to the LICENSE file included in the repository.
# -----------------------------------------------------------------------------

import socket
import cv2 as cv
from datetime import datetime


class Camera:
    def __init__(self, id: int, frame_width: int, frame_height: int, use: str):
        """
        Initializes the Camera object.

        :param id: int : The ID of the camera.
        :param frame_width: int : The width of the video frame.
        :param frame_height: int : The height of the video frame.
        :param use: str : The actual use of the camera.
        """
        self.__id = id  # ID of the camera
        self.use = use  # Actual use of the camera
        self.frame_width = frame_width  # Width of the video
        self.frame_height = frame_height  # Height of the video
        self.filming = False  # Flag for filming
        self.__cam = cv.VideoCapture(self.__id)

    def get_id(self) -> int:
        """
        Returns the ID of the camera.

        :return: int : The ID of the camera.
        """
        return self.__id

    def start_filming(self):
        """
        Starts filming with the camera.
        """
        if not self.filming:
            self.__cam.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.__cam.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.filming = True
            print("Starting filming.")
            self.__capture_frames()
        else:
            print("Already filming.")

    def stop_filming(self):
        """
        Stops filming with the camera.
        """
        if self.filming:
            self.filming = False
            self.__cam.release()
            self.__reload_cam()
            print("Filming stopped and camera released.")
        else:
            print("Camera is not filming.")

    def __reload_cam(self):
        """
        Reloads the camera capture object.
        """
        self.__cam = cv.VideoCapture(self.get_id())

    def __capture_frames(self):
        """
        Captures frames from the camera and sends them over UDP.
        """
        hostname = socket.gethostname()
        ip_to_send = socket.gethostbyname(hostname)
        port_to_send = 5001
        sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        while self.filming:
            ret, frame = self.__cam.read()
            if ret:
                try:
                    _, frame_encoded = cv.imencode(".jpg", frame)
                    sock.sendto(frame_encoded.tobytes(), (ip_to_send, port_to_send))
                except OSError as error:
                    size_sent_msg = len(frame_encoded.tobytes())
                    print("OsError:", error, "==>", size_sent_msg, "Bytes")
                self.check_fps()
                cv.imshow("Video sent", frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    self.stop_filming()
            else:
                print("Can't receive frame (stream end?). Exiting ...")
                self.filming = False
        cv.destroyAllWindows()

    def save_video(
        self, filename=datetime.strftime(datetime.now(), "%y_%m_%d__%H_%M_%S"), fs=30.0
    ):
        """
        Saves the video being filmed to a file.

        :param filename: str : The name of the file to save the video to. Default is the current datetime.
        :param fs: float : The frames per second of the saved video. Default is 30.0.
        """
        if self.filming:
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            out = cv.VideoWriter(
                "videos/" + filename + ".mp4",
                fourcc,
                fs,
                (self.frame_width, self.frame_height),
            )
            print(f'Saving video to "videos/{filename}.mp4"')
            self.__capture_frames(out)
            out.release()
        else:
            print("Trying to save a video while the camera isn't filming.")

    def check_camera(self) -> bool:
        """
        Checks if the camera is opened.

        :return: bool : True if the camera is opened, False otherwise.
        """
        if not self.__cam.isOpened():
            print("Error: Camera could not be opened.")
            return False
        return True

    def check_fps(self) -> float:
        """
        Checks the frames per second (FPS) of the camera.

        :return: float : The frames per second (FPS) of the camera.
        """
        fps = self.__cam.get(cv.CAP_PROP_FPS)
        return fps
