from camera import Camera
from models import MoveNet, MTCNN_Face

movenet = MoveNet("movenet")
mtcnn = MTCNN_Face("mtcnn")

cam = Camera(0, 640, 480, "", movenet)

cam.start_filming()
