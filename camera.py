import cv2 as cv
from datetime import datetime
from models import Model


class Camera:
    def __init__(
        self, id: int, frame_width: int, frame_height: int, use: str, model: Model
    ):
        self.__id = id  # ID's camera
        self.use = use  # Actual use of the camera
        self.frame_width = frame_width  # Width of the video
        self.frame_height = frame_height  # Height of the video
        self.model = model
        self.filming = False
        self.__cam = cv.VideoCapture(self.__id)

    def get_id(self):
        return self.__id

    def start_filming(self):
        if not self.filming:
            self.__cam.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.__cam.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.filming = True
            print("Starting filming.")
            self.__capture_frames()
        else:
            print("Already filming.")

    def stop_filming(self):
        if self.filming:
            self.filming = False
            self.__cam.release()
            self.__reload_cam()
            print("Filming stopped and camera released.")
        else:
            print("Camera is not filming.")

    def __reload_cam(self):
        self.__cam = cv.VideoCapture(self.get_id())

    def __capture_frames(self):
        while self.filming:
            ret, frame = self.__cam.read()
            if ret:
                # TOCAR AQUÍ frame PARA PROCESAR/ANALIZAR EL VÍDEO
                transformed_frame = self.model(frame)
                self.check_fps()
                cv.imshow("Video", transformed_frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    self.stop_filming()
            else:
                print("Can't receive frame (stream end?). Exiting ...")
                self.filming = False
        cv.destroyAllWindows()

    def save_video(
        self, filename=datetime.strftime(datetime.now(), "%y_%m_%d__%H_%M_%S"), fs=20.0
    ):
        if self.filming:
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            out = cv.VideoWriter(
                "videos/" + filename + ".mp4",
                fourcc,
                fs,
                (self.frame_width, self.frame_height),
            )
            print(f'Saving video to "videos/{filename}.mp4"')
            self.capture_frames(out)
            out.release()
        else:
            print("Trying to save a video while the camera isn't filming.")

    def check_camera(self):
        if not self.__cam.isOpened():
            print("Error: Camera could not be opened.")
            return False
        return True

    def check_fps(self):
        fps = self.__cam.get(cv.CAP_PROP_FPS)
        return fps
