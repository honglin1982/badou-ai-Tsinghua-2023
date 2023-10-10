#使用FRCNN模型来实时检测计算机捕获的视频流中的目标，并在窗口中显示检测结果
from keras.layers import Input
from frcnn import FRCNN
from PIL import Image
import numpy as np
import cv2

#创建FRCNN模型的实例
frcnn=FRCNN()

#打开计算机摄像头，0表示默认摄像头
capture=cv2.VideoCapture(0)

while True:
    #读取一帧视频
    ref,frame=capture.read()
    #将视频帧从BGR格式转换为RGB格式，适合后续处理
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #将帧数据转化为PIL图像格式，便于输入模型
    frame=Image.fromarray(np.uint8(frame))
    #使用FRCNN模型进行目标检测
    frame=np.array(frcnn.detect_image(frame))
    #将检测结果的图像格式从RGB转换为BGR，以便使用OpenCV显示
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    #在窗口中显示视频帧以其检测结果
    cv2.imshow("video",frame)
    #检查是否按下键盘上的Esc键（ASCII 为 27），如果是则停止循环
    c=cv2.waitKey(30)&0xff
    if c==27:
        capture.release()  #释放摄像头资源
        break
#关闭FRCNN模型会话，释放资源
frcnn.close_session()

