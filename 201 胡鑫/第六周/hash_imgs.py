import cv2
import os
import numpy as np
from PIL import ImageEnhance
from PIL import Image

def rotate(img):
    def rotate_bound(img, angle):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        # (cx, cy)旋转中心，angle旋转角度：正数表示逆时针，负数表示顺时针，单位为°，
        # 1.0表示缩放系数，默认为1.0，不进行缩放
        # M.shape = (2, 3)
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        # print(M)
        # 前两个为余弦和正弦，取绝对值
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # 计算新边界维度
        nw = int((h * sin) + (w * cos))
        nh = int((h * cos) + (w * sin))

        # 更新偏移量
        M[0, 2] += (nw / 2) - cx
        M[1, 2] += (nh / 2) - cy
        # 根据变换矩阵M将每个像素点映射到新位置
        return cv2.warpAffine(img, M, (nw, nh))
    return rotate_bound(img, 45)

def color(img):
    # enh_color = ImageEnhance.Color(img) 这行代码将 img 作为参数创建了一个 enh_color 对象；
    # 其中 Color() 方法用于增强图像的颜色饱和度，它返回一个 ImageEnhance 类的实例对象，
    # 可以调用其中的 enhance() 方法进一步增强图像的颜色饱和度。
    enh_color = ImageEnhance.Color(img)
    # 增强1.5倍
    color = 1.5
    return enh_color.enhance(color)

def blur(img):
    # 对图像进行均值滤波，滤波器形状（15， 1）
    return cv2.blur(img, (15, 1))

def sharp(img):
    # laplace锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    # -1表示输出图像与原图像的数据类型相同，否则指定数据类型，例如cv.CV_8U等等
    return cv2.filter2D(img, -1, kernel=kernel)

def constrast(img):
    def constrast_img(src1, a, g):
        """粗略调整对比度和亮度

        Args:
            src1 (ndarray): 图像1
            a (float): 比率
            g (float): 亮度
        """
        h, w, c = src1.shape
        # 新建一个全黑的图像，与src大小、通道数全部相同
        src2 = np.zeros( (h, w, c), dtype=src1.dtype )
        # 调整
        # addWeighted是将两个图像按权重合并
        # src1为第一个图像，a为第二个图像的权重，第二三个参数同理，第五个参数为亮度
        # 公式： src1 * a + src2 * （1-a） + g
        return cv2.addWeighted(src1, a, src2, 1-a, g)
    return constrast_img(img, 1.2, 1)

def resize(img):
    # (0, 0)表示按照后面的比例自动缩放
    return cv2.resize(img, (0, 0), fx=1.25, fy=1)

def light(img):
    # 通过np.clip()怎么像素亮度，0和255限制像素值
    return np.uint8(np.clip(1.3 * img + 10, 0, 255))

def save_img(img, file_name, path):
    # 写入时采用 JPEG 格式，并设置压缩质量为 70。
    cv2.imwrite(os.path.join(path, file_name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

def show(img):
    print(img.shape)
    cv2.imshow('img_rotate', img)
    cv2.waitKey(0)

def main():
    data_img_name = 'lenna.png'
    output_path = './source'
    name = data_img_name.split(".")[0]
    
    data_path = '../lenna.png'
    img = cv2.imread(data_path)

    '''通过以下处理分别得出不同情况的相似图片'''
    # 修改图片的亮度
    img_light = light(img)
    # 修改图片的大小
    img_resize = resize(img)
    # 修改图片的对比度
    img_constrast = constrast(img)
    # 锐化图片
    img_sharp = sharp(img)
    # 模糊
    img_blur = blur(img)
    # 色度增强
    img_color = color(Image.open(data_path))
    # 图像旋转
    img_rotate = rotate(img)
    img_rotate1 = Image.open(data_path).rotate(45)

    # 储存图像
    save_img(img_light, f'{name}_light.jpg', output_path)
    save_img(img_resize, f'{name}_resize.jpg', output_path)
    save_img(img_constrast, f'{name}_constrast.jpg', output_path)
    save_img(img_sharp, f'{name}_sharp.jpg', output_path)
    save_img(img_blur, f'{name}_blur.jpg', output_path)

    img_color.save(os.path.join(output_path, f'{name}_color.jpg'))
    img_rotate1.save(os.path.join(output_path, f'{name}_rotate1.jpg'))

    show(img_rotate)

if __name__ == "__main__":
    main()