import cv2
from matplotlib import pyplot as plt
import numpy as np


class Week03:

    def __init__(self, path=None):
        self.img = cv2.imread(path)

    def base_nearest_neighbor_interpolation_method(self, target_h, target_w):
        """
        底层原理代码展示
        :param target_h: 目标图片的长度
        :param target_w: 目标图片的宽度
        :return:
        """
        # 最邻近插值法
        h, w, channels = self.img.shape
        new_img = np.zeros((target_h, target_w, channels), dtype=np.uint8)
        rate_h = target_h / h
        rate_w = target_w / w
        for i in range(target_h):
            for j in range(target_w):
                x = int(i / rate_h + 0.5)
                y = int(j / rate_w + 0.5)
                if x >= h:
                    x = h - 1
                if y >= w:
                    y = w - 1
                # print(f'{new_img[i, j]} --{x, y}-> {img[x, y]}')
                new_img[i, j] = self.img[x, y]
        # print(new_img)
        # print(new_img.shape)
        cv2.imshow("nearest interp", new_img)
        cv2.imshow("image", self.img)
        cv2.waitKey(10000)

    def cv2_nearest_neighbor_interpolation_method(self, target_h, target_w):
        """
        cv2 中对于最邻近插值法写法
        :param target_h: 目标图片的长度
        :param target_w: 目标图片的宽度
        cv2.resize：进行图片的缩放行为作用
        :return:
        """
        # 放大,双立方插值
        # new_img = cv2.resize(img, (target_h, target_w), interpolation=cv2.INTER_NEAREST)
        # 放大, 象素关系重采样
        # new_img = cv2.resize(img, (target_h, target_w), interpolation=cv2.INTER_CUBIC)
        # 缩小, 象素关系重采样
        # new_img = cv2.resize(img, (300, 200), interpolation=cv2.INTER_AREA)
        # 放大, 最近邻插值
        new_img = cv2.resize(self.img, (target_h, target_w), interpolation=cv2.INTER_AREA)
        cv2.imshow("nearest interp", new_img)
        cv2.imshow("image", self.img)
        cv2.waitKey(10000)

    def bilinear_interpolation(self, source_img: np.ndarray, target_h: int, target_w: int):
        """
        形状image.shape： H*W*C
        索引：image[y][x]
        通道顺序：BGR
        """
        src_h, src_w, channels = source_img.shape
        if src_h == target_h and src_w == target_w:
            return source_img.copy()
        dst_img = np.zeros((target_h, target_w, channels), dtype=np.uint8)
        scale_y = src_h / target_h
        scale_x = src_w / target_w
        for i in range(channels):
            for dst_y in range(target_h):
                for dst_x in range(target_w):
                    # 利用目标图像的xy求出原图像虚拟点位置
                    virtual_src_y = (dst_y + 0.5) * scale_y - 0.5
                    virtual_src_x = (dst_x + 0.5) * scale_x - 0.5
                    # 再求出虚拟点周围实际的四个点的坐标数值，注：只需要求出两个坐标值就行 （x0, y0）, (x1, y1)
                    src_x0 = int(np.floor(virtual_src_x))
                    src_y0 = int(np.floor(virtual_src_y))
                    src_x1 = min(src_x0 + 1, src_w - 1)
                    src_y1 = min(src_y0 + 1, src_h - 1)
                    temp0 = (src_x1 - virtual_src_x) * source_img[src_y0, src_x0, i] + (virtual_src_x - src_x0) * \
                            source_img[src_y0, src_x1, i]
                    temp1 = (src_x1 - virtual_src_x) * source_img[src_y1, src_x0, i] + (virtual_src_x - src_x0) * \
                            source_img[src_y1, src_x1, i]
                    dst_img[dst_y, dst_x, i] = (src_y1 - virtual_src_y) * temp0 + (virtual_src_y - src_y0) * temp1
        print(dst_img)
        print(dst_img.shape)
        cv2.imshow("dst_img", dst_img)
        cv2.imshow("source_img", source_img)
        cv2.waitKey(20000)

    def single_histogram_equalization(self):
        """ 单通道灰度直方图均衡化 """
        img_gray = cv2.cvtColor(self.img, code=cv2.COLOR_BGR2GRAY)
        dst = cv2.equalizeHist(img_gray)  # 直方图均衡化
        print(dst)
        cv2.imshow(" histogram_equalization ", np.hstack([img_gray, dst]))
        cv2.waitKey(5000)
        # plt 直方图展示方法描述每个灰度级出现的次数
        plt.figure()
        plt.hist(dst.ravel(), 256)
        plt.show()
        # cv2 直方图描述每个灰度级出现的次数
        hist = cv2.calcHist([dst], [0], None, [256], [0, 255])

    def three_single_histogram_equalization(self):
        """ 三通道灰度直方图均衡化 """
        b, g, r = cv2.split(self.img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        dst_img = cv2.merge((bH, gH, rH))
        cv2.imshow(" rgb dst ", np.hstack([self.img, dst_img]))
        cv2.waitKey(5000)

        # # 直方图展示方法
        # plt.figure()
        # plt.hist(self.img.ravel(), 256)
        # print(dst_img.ravel())
        # plt.hist(dst_img.ravel(), 256)
        # plt.show()


if __name__ == '__main__':
    obj = Week03(path=r'../file/lenna.png')
    # obj.base_nearest_neighbor_interpolation_method(900, 900)  # 邻近插值源码分析
    # obj.cv2_nearest_neighbor_interpolation_method(900, 900)  # cv2的邻近插值方法
    # obj.bilinear_interpolation(obj.img, target_h=900, target_w=900)  # 双线性插值源码分析
    # obj.single_histogram_equalization()  # 单通道灰度直方图均衡化
    obj.three_single_histogram_equalization()  # 三通道灰度直方图均衡化
    pass


