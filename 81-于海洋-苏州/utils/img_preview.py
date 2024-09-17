from matplotlib import pyplot as plt
from numpy import ndarray


class ImgPreview:
    index: int = 0

    def __init__(self, width: int, height: int, row: int, column: int, ):
        self.row = row
        self.column = column
        plt.figure(figsize=(width, height))

    def gray(self, img: ndarray, title=''):
        self.add(img, title, grey=True)

    def add(self, img: ndarray, title='', grey: bool = False):
        """
        cmap: 有以下枚举值。 默认值为None
        viridis：这是一种常用的颜色映射，它从深蓝色到亮黄色表示较低到较高的值。
        gray：这是一种灰度颜色映射，用于将数值转换为不同的灰度级别。
        hot：这种颜色映射以黑色为基准，通过红、橙、黄、白的渐变来表示较低到较高的值。
        cool：这种颜色映射从青色到紫色表示较低到较高的值。
        """

        self.index = self.index + 1

        if grey:
            camp = "gray"
        else:
            camp = None

        plt.subplot(self.row, self.column, self.index)
        plt.title(title)
        plt.imshow(img, cmap=camp)

    @staticmethod
    def show():
        plt.show()
