import  os
import  config
import  numpy as np
import  tensorflow as tf
from yoloPredict import  yoloPredictor
from PIL import  Image, ImageFont, ImageDraw
from utils import  letterboxImage, loadWeights

#指定使用GPU的Index
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpuIndx

def detect(imagePath, modelPath, yoloWeights = None):
    """
       Introduction
       ------------
           加载模型，进行预测
    """
    #预处理图片
    image = Image.open(imagePath)
    #对预测输入图像进行缩放
    resizeImage = letterboxImage(image, (416, 416))
    imageData = np.array(resizeImage, dtype = np.float32)
    #归一化
    imageData /= 255.
    #转格式，第一维度填充
    imageData = np.expand_dims(imageData, axis = 0)
    #图片输入
    inputImageShape = tf.placeholder(dtype = tf.int32, shape = (2,))
    #图像
    inputImage = tf.placeholder(shape = [None, 416, 416, 3], dtype = tf.float32)

    # 进入yoloPredictor进行预测，yoloPredictor是用于预测的一个对象
    predictor = yoloPredictor(config.objThreshold, config.nmsThreshold,
                              config.classesPath, config.anchorsPath)
    with tf.Session() as sess:
        # 图片预测
        if yoloWeights is not  None :
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(inputImage, inputImageShape)
            #载入模型
            loadOp = loadWeights(tf.global_variables(scope = 'predict'), weightsFile = yoloWeights)
            sess.run(loadOp)

            #进行预测
            outBoxes, outScores, outClasses = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    inputImage : imageData,
                    inputImageShape : [image.size[1], image.size[0]]
                }
            )
        else :
            boxes, scores, classes = predictor.predict(inputImage, inputImageShape)
            saver = tf.train.Saver()
            saver.restore(sess, modelPath)
            outBoxes, outScores, outClasses = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    inputImage : imageData,
                    inputImageShape : [image.size[1], image.size[0]]
                }
            )
        # 画框
        print('Found {} boxes for {}'.format(len(outBoxes), 'img'))
        font = ImageFont.truetype(font = 'font/FiraMono-Medium.otf',
                                  size = np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        #厚度
        thickness = (image.size[0] + image.size[1]) // 300
        for i, c in reversed(list(enumerate(outClasses))) :
             # 获得预测名字，box和分数
             predictedClass = predictor.classesNames[c]
             box = outBoxes[i]
             score = outScores[i]

             #打印
             label = '{} {:.2f}'.format(predictedClass, score)
             # 用于画框框和文字
             draw = ImageDraw.Draw(image)
             # textsize用于获得写字的时候，按照这个字体
             labelSize = draw.textsize(label, font)

             #获得四个边
             top, left, bottom, right = box
             top = max(0, np.floor(top + 0.5).astype('int32'))
             left = max(0, np.floor(left + 0.5).astype('int32'))
             bottom = min(image.size[1] - 1, np.floor(bottom + 0.5).astype('int32'))
             right = min(image.size[0] - 1, np.floor(right + 0.5).astype('int32'))
             print(label, (left, top), (right, bottom))
             print(labelSize)

             if top - labelSize[1] >= 0: textOrigin = np.array([left, top - labelSize[1]])
             else: textOrigin = np.array([left, top + 1])

             for i in range(thickness):
                 draw.rectangle([left + i, top + i, right - i, bottom - i], outline = predictor.colors[c])
             draw.rectangle([tuple(textOrigin), tuple(textOrigin + labelSize)], fill = predictor.colors[c])
             draw.text(textOrigin, label, fill = (0, 0, 0), font = font)
             del draw
        image.show()
        image.save('./img/result1.jpg')

if __name__ == '__main__':

    if config.preTrainYolo3 == True : detect(config.imageFile, config.model_dir, config.yolo3WeightsPath)
    else : detect(config.imageFile, config.model_dir)