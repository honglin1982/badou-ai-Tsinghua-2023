from frcnn import FRCNN
from PIL import Image
import numpy as np
import cv2


frcnn = FRCNN()

capture = cv2.VideoCapture(0)

while True:
    ref, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(np.uint8(frame))
    frame = np.array(frcnn.detect_image(frame))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('video', frame)
    c = cv2.waitKey(30) & 0xff
    if c == 27:
        capture.release()
        break
frcnn.close_session()
