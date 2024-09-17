import cv2
import numpy as np
from skimage import util

'''
def random_noisy(image, mode='', seed=None, clip=True, **kwargs):
功能: 为浮点型图片添加各种随机噪声
参数:
image: 输入的图片(将会被转换成浮点型), ndarray型
mode: 可选择, str型, 表示要添加的噪声类型
    gaussian: 高斯噪声
    localvar: 高斯分布的加性噪声, 在"图像"的每个点处, 具有指定的局部方差
    poisson: 泊松噪声
    salt: 盐噪声, 随机将像素值变成1
    pepper: 椒噪声, 随机将像素的值变成0或者-1, 取决于矩阵的值是否带符号
    s&p: 椒盐噪声
    speckle: 均匀噪声(均值为mean方差为variance), out = image + n*image
seed: 可选的, int型, 如果选择的话, 在生成噪声前会先设置随机种子以避免伪随机
clip: 可选的, bool型, 如果是True, 在添加均值, 泊松以及高斯噪声后, 会将图片
      数据裁剪到合适范围内. 如果设为False, 则输出矩阵的值可能会超过[-1, 1]
mean: 可选的, float型, 均值噪声和高斯噪声中的mean参数, 默认为0
var: 可选的, float型, 均值噪声和高斯噪声中的方差, 默认值=0.01(注:不是标准差)
local_vars: 可选的, ndarray型, 用于定义每个像素点的局部方差, localvar中使用
amount: 可选的, float型, 是椒盐噪声所占比例, 默认值=0.05
salt_vs_pepper: 可选的, float型, 椒盐噪声中椒盐比例, 值越大表示盐噪声越多, 
                默认值=0.5, 即椒盐等量
--------
返回值: ndarray型, 且值在[0, 1]或[-1, 1]之间, 却决于是否有符号数               
'''

img = cv2.imread('../lenna.png')
img1 = util.random_noise(img, mode='gaussian', mean=0, var=0.01, clip=True, seed=1)

img = cv2.imread('../lenna.png')
cv2.imshow('source', img)
cv2.imshow('poisson', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()