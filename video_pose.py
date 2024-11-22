from models import MoveNet
import cv2 as cv
from datetime import time, timedelta
import json

"""
Violence: 20, 57, 520, 740, 747, 836, 845, 860, 944, 957
NonViolence: 
"""

validation = True
violence = False
num = 1
folder = f'{"" if violence else "Non"}Violence' if not validation else "Validation"
video_name = (
    f'{"V" if violence else "NV"}_{num}.mp4'
    if not validation
    else f"bullying_{num}.mp4"
)

cap = cv.VideoCapture(f"train_videos/{folder}/{video_name}")
len_video = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
model = MoveNet(
    "", int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), 0
)

t_init = time(minute=0, second=49)
td_init = int(
    timedelta(
        hours=t_init.hour, minutes=t_init.minute, seconds=t_init.second
    ).total_seconds()
)
starting_frame = td_init * cap.get(cv.CAP_PROP_FPS)

t_end = time(minute=0, second=53)
td_end = int(
    timedelta(
        hours=t_end.hour, minutes=t_end.minute, seconds=t_end.second
    ).total_seconds()
)
final_frame = td_end * cap.get(cv.CAP_PROP_FPS)

dataset = list()
cap.set(cv.CAP_PROP_POS_FRAMES, starting_frame)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        trans_frame = model(frame)
        cv.imshow("video", trans_frame)
        nf = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        preds = model.predict(model.preprocess(frame))
        dataset.append(preds)
        if (cv.waitKey(25) & 0xFF == ord("q")) or int(
            cap.get(cv.CAP_PROP_POS_FRAMES)
        ) + 1 == final_frame:
            break
    else:
        break

with open(f"poses/{folder}/poses_{num}.json", "wt") as new_file:
    json.dump(dataset, new_file)
    print(f"Saved to: poses/{folder}/poses_{num}.json")

cap.release()
cv.destroyAllWindows()
