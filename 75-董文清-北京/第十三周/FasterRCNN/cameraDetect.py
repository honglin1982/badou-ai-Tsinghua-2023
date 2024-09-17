from keras.layers import  Input
from frcnn import  FRCNN
from PIL import Image
import numpy as np
import cv2

frcnn = FRCNN()

#调用摄像头
capture = cv2.VideoCapture(0)
while(True):
    #读取一帧
    ref, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    frame = Image.fromarray(np.uint8(frame))

    #进行检测
    frame = np.array(frcnn.detectImage(frame))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    cv2.imshow("video", frame)
    c = cv2.waitKey(30) & 0xff
    if c == 27:
        capture.release()
        break

frcnn.closeSession()