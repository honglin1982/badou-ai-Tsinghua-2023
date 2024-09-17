import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


class CannyAlgorithmPrinciple(object):
    """
    Canny 算法
    过程原理:
    1.高斯平滑, 求出卷积核
    2.填充 + 高斯卷积得出卷积后的图像
    3.sobel算子求出边缘梯度
    4.非极大值抑制
    5.双阈值检测，连接边缘
    """

    def __init__(self, pic_path=None):
        self.pic_path = pic_path
        self.img_gray = self.get_img_gray()
        self.gaussian_filter = None  # 高斯平滑,
        self.dim = None  # 卷积核大小
        self.img_new = None  # 填充 + 高斯卷积的新图像
        self.angle = None  # sobel算子求出tan 值图像
        self.img_tidu = None  # sobel算子求出边缘梯度图像
        self.img_yizhi = None  # 非极大值抑制求出的图像

    @property
    def sigma(self):
        """ 高斯sigma """
        return 0.5

    @property
    def dimensionality(self):
        """ 自定义的 卷积核大小 """
        return 5

    @staticmethod
    def padding(img: np.ndarray, params, mod="constant") -> np.ndarray:
        """ 边缘填补 """
        img_pad = np.pad(img, params, mod)  # 边缘填补
        return img_pad

    def get_img_gray(self) -> np.ndarray:
        img = plt.imread(self.pic_path)
        if self.pic_path[-4:] == ".png":
            img = img * 255
        img_gray = img.mean(axis=-1)  # 取均值就是灰度化
        return img_gray

    def get_gaussian_filter(self):
        """1、高斯平滑"""
        # sigma = 1.52  # 高斯平滑时的高斯核参数，标准差，可调
        # sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
        dim = int(np.round(6 * self.sigma + 1))  # round是四舍五入函数，根据标准差求高斯核是几乘几的，也就是维度
        if dim % 2 == 0:  # 最好是奇数,不是的话加一
            dim += 1
        Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了
        tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列
        # print(f'---tmp: {tmp}')  # [-2, -1, 0, 1, 2]
        n1 = 1 / (2 * math.pi * self.sigma ** 2)  # 计算高斯核
        n2 = -1 / (2 * self.sigma ** 2)
        for i in range(dim):
            for j in range(dim):
                Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
        Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
        assert self.dimensionality == dim
        return Gaussian_filter, dim

    def padding_img(self):
        """ 填充 + 高斯卷积 """
        dx, dy = self.img_gray.shape
        img_new = np.zeros(self.img_gray.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
        tmp = self.dim // 2  # 填充的维度数
        img_pad = self.padding(self.img_gray, ((tmp, tmp), (tmp, tmp)), mod="constant")  # 边缘填补
        for i in range(dx):
            for j in range(dy):
                img_new[i, j] = np.sum(img_pad[i: i + self.dim, j: j + self.dim] * self.gaussian_filter)
        # print('高斯卷积核卷积结果：')
        # print(img_new)
        # 尺寸和原img一样，因为卷积前padding了
        # 画图1
        plt.figure(1)
        plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
        plt.axis('off')
        return img_new

    def sobel_kernel_func(self):
        """ sobel算子求出边缘 """
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        # dx, dy = self.img_new.shape
        img_tidu_x = np.zeros(self.img_new.shape)  # 存储梯度图像
        # img_tidu_y = np.zeros([dx, dy])
        img_tidu_y = np.zeros(self.img_new.shape)  # 因为 卷积后的图尺寸大小一样的
        img_tidu = np.zeros(self.img_new.shape)
        img_pad = np.pad(self.img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
        for i in range(self.img_new.shape[0]):
            for j in range(self.img_new.shape[1]):
                img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向
                img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向
                # todo 梯度是向量的，在向量空间中向量的模就等于x、y的平方和再开根号
                img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
        img_tidu_x[img_tidu_x == 0] = 0.00000001
        angle = img_tidu_y / img_tidu_x  # tan 数值图像 y/x
        plt.figure(2)
        # sobel 算子的出来的 梯度边缘
        plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
        plt.axis('off')
        return angle, img_tidu

    def non_maximum_suppression(self):
        '''
        3、非极大值抑制
        angle[i, j] 其实就是当前梯度的方向，img_tidu[i, j]是当前梯度值，利用梯度幅值的8邻域矩阵
        求的就是梯度值在该方向上是否是最大的
        '''
        dx, dy = self.img_tidu.shape[0], self.img_tidu.shape[1]
        img_yizhi = np.zeros((dx, dy))
        for i in range(1, dx - 1):
            for j in range(1, dy - 1):
                # 在8邻域内是否要抹去做个标记
                temp = self.img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
                if self.angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                    num_1 = (temp[0, 1] - temp[0, 0]) / self.angle[i, j] + temp[0, 1]
                    num_2 = (temp[2, 1] - temp[2, 2]) / self.angle[i, j] + temp[2, 1]
                elif self.angle[i, j] >= 1:
                    num_1 = (temp[0, 2] - temp[0, 1]) / self.angle[i, j] + temp[0, 1]
                    num_2 = (temp[2, 0] - temp[2, 1]) / self.angle[i, j] + temp[2, 1]
                elif self.angle[i, j] > 0:
                    num_1 = (temp[0, 2] - temp[1, 2]) * self.angle[i, j] + temp[1, 2]
                    num_2 = (temp[2, 0] - temp[1, 0]) * self.angle[i, j] + temp[1, 0]
                elif self.angle[i, j] < 0:
                    num_1 = (temp[1, 0] - temp[0, 0]) * self.angle[i, j] + temp[1, 0]
                    num_2 = (temp[1, 2] - temp[2, 2]) * self.angle[i, j] + temp[1, 2]
                else:
                    continue
                if self.img_tidu[i, j] > num_1 and self.img_tidu[i, j] > num_2:
                    img_yizhi[i, j] = self.img_tidu[i, j]
        plt.figure(3)
        plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
        plt.axis('off')
        return img_yizhi

    def dual_threshold_detection_connecting_edges(self):
        """
        4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈

        当前被标记为强边缘的位置的梯度值，附近8个领域点，如果这个8个点其中任意的梯度值处于
        弱边缘像素点的（说明这个弱边缘像素点事能够连接到强边缘的，也就是该点也属于边缘像素点），
        将其视为我们要取的边缘点，并对其进行下一步追踪它的8领域点进行判断
        """
        lower_boundary = self.img_tidu.mean() * 0.5
        high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
        zhan = []  # 存储需要标记的强边缘位置
        for i in range(1, self.img_yizhi.shape[0] - 1):  # 外圈不考虑了
            for j in range(1, self.img_yizhi.shape[1] - 1):
                if self.img_yizhi[i, j] <= lower_boundary:
                    self.img_yizhi[i, j] = 0
                elif self.img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                    self.img_yizhi[i, j] = 255
                    zhan.append([i, j])  # 存储强度边缘梯度位置
        while len(zhan) > 0:
            temp_1, temp_2 = zhan.pop()  # 出栈
            a = self.img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
            if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
                self.img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
                zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
            if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
                self.img_yizhi[temp_1 - 1, temp_2] = 255
                zhan.append([temp_1 - 1, temp_2])
            if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
                self.img_yizhi[temp_1 - 1, temp_2 + 1] = 255
                zhan.append([temp_1 - 1, temp_2 + 1])
            if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
                self.img_yizhi[temp_1, temp_2 - 1] = 255
                zhan.append([temp_1, temp_2 - 1])
            if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
                self.img_yizhi[temp_1, temp_2 + 1] = 255
                zhan.append([temp_1, temp_2 + 1])
            if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
                self.img_yizhi[temp_1 + 1, temp_2 - 1] = 255
                zhan.append([temp_1 + 1, temp_2 - 1])
            if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
                self.img_yizhi[temp_1 + 1, temp_2] = 255
                zhan.append([temp_1 + 1, temp_2])
            if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
                self.img_yizhi[temp_1 + 1, temp_2 + 1] = 255
                zhan.append([temp_1 + 1, temp_2 + 1])

        for i in range(self.img_yizhi.shape[0]):
            for j in range(self.img_yizhi.shape[1]):
                if self.img_yizhi[i, j] != 0 and self.img_yizhi[i, j] != 255:
                    self.img_yizhi[i, j] = 0
        # 绘图
        plt.figure(4)
        plt.imshow(self.img_yizhi.astype(np.uint8), cmap='gray')
        plt.axis('off')  # 关闭坐标刻度值
        plt.show()

    def run_fit(self):
        self.gaussian_filter, self.dim = self.get_gaussian_filter()  # 高斯平滑, 卷积核大小
        self.img_new = self.padding_img()  # 填充 + 高斯卷积
        self.angle, self.img_tidu = self.sobel_kernel_func()  # sobel算子求出边缘梯度
        self.img_yizhi = self.non_maximum_suppression()  # 非极大值抑制
        self.dual_threshold_detection_connecting_edges()  # 双阈值检测，连接边缘


if __name__ == '__main__':
    pic_path = r'../file/lenna.png'
    cap_obj = CannyAlgorithmPrinciple(pic_path)
    cap_obj.run_fit()
    plt.show()





