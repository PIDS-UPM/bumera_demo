from modules.models import MoveNet
import json
import cv2 as cv
from datetime import datetime

"""
Violence: 20, 57, 520, 740, 747, 836, 845, 860, 944, 957
NonViolence: 
"""

for i in range(101, 1001):
    ini = datetime.now()
    violence = True
    num = i
    video_name = f'{"V" if violence else "NV"}_{num}.mp4'
    cap = cv.VideoCapture(
        f'train_videos/{"" if violence else "Non"}Violence/{video_name}'
    )
    # out = cv.VideoWriter("videos/output.mp4", cv.VideoWriter_fourcc(*"mp4v"), 20.0, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    model = MoveNet(
        "MoveNet",
        int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        0,
    )

    dataset = list()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # trans_frame = model(frame) # VA MAL POR EL TAMAÑO DEL INPUT, QUE DENTRO HACE MAL LA TRANSFORMACIÓN
            # cv.imshow('video', trans_frame)
            nf = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            preds = model.predict(model.preprocess(frame))
            dataset.append(preds)
            if cv.waitKey(25) & 0xFF == ord("q"):
                break
        else:
            with open(
                f'poses/{"" if violence else "Non"}Violence/poses_{num}.json', "wt"
            ) as new_file:
                json.dump(dataset, new_file)
            break
    print(f"{num}) {(datetime.now()-ini).seconds} segundos ==> {len(dataset)}")

    cap.release()
    cv.destroyAllWindows()
