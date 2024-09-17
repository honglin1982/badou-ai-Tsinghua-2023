import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


# 单通道图像 K-Means聚类
def kmeans_1():
    img = cv2.imread("lenna.png", 0)   # 0 灰度模式读取
    rows, cols = img.shape[:]  # 获取行列
    data = img.reshape((rows * cols, 1))  # 改变数组结构（512*512 行，1 列 ）
    data = np.float32(data)  # 转换数据类型
    # 停止条件设置: 最大迭代次数 10， 精度阈值 1.0， 满足任意一个就停止
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 设置初始中心的选择方式：随机选择初始中心（cv2.KMEANS_RANDOM_CENTERS）
    flags = cv2.KMEANS_RANDOM_CENTERS
    # 使用K-Means聚类，聚集成4类，重复执行10次，每次的初始化中心不同
    compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
    print(centers)
    # 生成最终图像:labels是一个整数数组，存储了每个样本的聚类标签索引，表示每个样本被分配到哪个聚类中心。
    # dst_img = labels.reshape((rows, cols))   # 修改数组结构
    centers = np.uint8(centers)
    dst_img = centers[labels.flatten()]
    dst_img = dst_img.reshape(img.shape)  # 4值图
    print(dst_img)

    # 正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图像
    titles = ["原始图像", "聚类图像"]
    images = [img, dst_img]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        # 尽管聚类标签dst_img的值是整数（0、1、2、3等）, 但imshow()函数显示灰度图像时，默认会根据像素值的范围自动进行灰度映射。具体来说，它会将最小值映射为黑色，最大值映射为白色，中间的值进行线性映射。
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])  # 隐藏x轴刻度。
        plt.yticks([])  # 隐藏y轴刻度。
    plt.show()

# 多通道图像 K-Means聚类
def kmeans_2():
    img = cv2.imread("lenna.png")
    print(img.shape)
    # -1 表示自动计算行数，保持图像的总像素数不变，而 3 表示每个像素点的颜色通道数量
    # 每行表示一个像素点，每列表示一个颜色通道的数值, 3 就有 3列， 行数为 图像的总像素 512*512
    data = img.reshape((-1, 3))
    data = np.float32(data)
    print(data.shape)
    # 停止条件设置: 最大迭代次数 10， 精度阈值 1.0， 满足任意一个就停止
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 设置初始中心的选择方式：随机选择初始中心（cv2.KMEANS_RANDOM_CENTERS）
    flags = cv2.KMEANS_RANDOM_CENTERS

    # K-Means聚类 聚集成2类
    compactness2, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
    # K-Means聚类 聚集成4类
    compactness4, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
    # # K-Means聚类 聚集成8类
    compactness8, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)
    # # K-Means聚类 聚集成16类
    compactness16, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)
    # # K-Means聚类 聚集成32类
    compactness32, labels32, centers32 = cv2.kmeans(data, 32, None, criteria, 10, flags)
    # # K-Means聚类 聚集成64类
    compactness, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

    # 图像转换回uint8二维类型，并将聚类标签索引展平为一维数组，并获取对应中心点的值，转换结构
    centers2 = np.uint8(centers2)
    dst_2 = centers2[labels2.flatten()]
    dst_2 = dst_2.reshape(img.shape)   # 2值图

    centers4 = np.uint8(centers4)
    dst_4 = centers4[labels4.flatten()]
    dst_4 = dst_4.reshape(img.shape)  # 4值图

    centers8 = np.uint8(centers8)
    dst_8 = centers8[labels8.flatten()]
    dst_8 = dst_8.reshape(img.shape)  # 8值图

    centers16 = np.uint8(centers16)
    dst_16 = centers16[labels16.flatten()]
    dst_16 = dst_16.reshape(img.shape)  # 16值图

    centers32 = np.uint8(centers32)
    dst_32 = centers32[labels32.flatten()]
    dst_32 = dst_32.reshape(img.shape)  # 32值图

    centers64 = np.uint8(centers64)
    dst_64 = centers64[labels64.flatten()]
    dst_64 = dst_64.reshape(img.shape)  # 64值图

    # 显示图像，以RGB的形式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst_2 = cv2.cvtColor(dst_2, cv2.COLOR_BGR2RGB)
    dst_4 = cv2.cvtColor(dst_4, cv2.COLOR_BGR2RGB)
    dst_8 = cv2.cvtColor(dst_8, cv2.COLOR_BGR2RGB)
    dst_16 = cv2.cvtColor(dst_16, cv2.COLOR_BGR2RGB)
    dst_32 = cv2.cvtColor(dst_32, cv2.COLOR_BGR2RGB)
    dst_64 = cv2.cvtColor(dst_64, cv2.COLOR_BGR2RGB)

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 显示图像
    titles = ['原始图像', '聚类图像 K=2', '聚类图像 K=4',
              '聚类图像 K=8', '聚类图像 K=16', '聚类图像 K=32', '聚类图像 K=64']
    images = [img, dst_2, dst_4, dst_8, dst_16, dst_32, dst_64]
    for i in range(7):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# 使用sklearn库实现kmeans聚类
def sklearn_kmeans():
    """
    第一部分：数据集
    X表示二维矩阵数据，篮球运动员比赛数据
    总共20行，每行两列数据
    第一列表示球员每分钟助攻数：assists_per_minute
    第二列表示球员每分钟得分数：points_per_minute
    """
    X = [[0.0888, 0.5885],
         [0.1399, 0.8291],
         [0.0747, 0.4974],
         [0.0983, 0.5772],
         [0.1276, 0.5703],
         [0.1671, 0.5835],
         [0.1306, 0.5276],
         [0.1061, 0.5523],
         [0.2446, 0.4007],
         [0.1670, 0.4770],
         [0.2485, 0.4313],
         [0.1227, 0.4909],
         [0.1240, 0.5668],
         [0.1461, 0.5113],
         [0.2315, 0.3788],
         [0.0494, 0.5590],
         [0.1107, 0.4799],
         [0.1121, 0.5735],
         [0.1007, 0.6318],
         [0.2567, 0.4326],
         [0.1956, 0.4280]
         ]
    # 输出数据集
    print(X)

    """
    第二部分：KMeans聚类
    clf = KMeans(n_clusters=3) 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
    y_pred = clf.fit_predict(X) 载入数据集X，并且将聚类的结果赋值给y_pred
    """
    X = np.array(X)
    clf = KMeans(n_clusters=3)
    y_pred = clf.fit_predict(X)
    # 输出完整Kmeans函数，包括很多省略参数
    print(clf)
    # 输出聚类预测结果
    print("y_pred = ", y_pred)

    """
    第三部分：可视化绘图 
    绘制散点图 
    参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
    """
    # 绘制每个类别的散点图并设置图例
    labels = ["A", "B", "C"]
    for i in range(3):
        plt.scatter(X[y_pred == i, 0], X[y_pred == i, 1], marker="x", label=labels[i])
    # 绘制标题
    plt.title("Kmeans-Basketball Data")
    # 绘制x轴和y轴坐标
    plt.xlabel("assists_per_minute")
    plt.ylabel("points_per_minute")
    # 设置右上角图例
    plt.legend()
    # 显示图形
    plt.show()


if __name__ == '__main__':
    kmeans_1()  # 单通道图像聚类
    kmeans_2()  # 多通道图像聚类
    sklearn_kmeans()  # sklearn 实现KMeans聚类
    