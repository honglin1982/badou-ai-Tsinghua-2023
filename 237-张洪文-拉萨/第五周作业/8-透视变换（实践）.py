import cv2
import numpy as np

"""
1、定义源图像中感兴趣的区域的四个顶点。
2、定义目标图像中对应的四个顶点。
3、使用cv2.getPerspectiveTransform()函数计算透视变换矩阵，该矩阵描述了源图像到目标图像的转换关系。
4、将透视变换矩阵和源图像传递给cv2.warpPerspective()函数，进行透视变换。
5、得到经过透视变换后的图像。
"""

# 透视变换 opencv 接口调用
def perspective_transformation(img):
    # 源图像的4个顶点
    src_points = np.float32([[50, 50], [200, 50], [200, 200], [50, 200]])
    # 目标图像中对应的4个顶点
    dst_points = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    # 计算透视变换矩阵
    m = cv2.getPerspectiveTransform(src_points, dst_points)
    print(f"透视变换矩阵: {m}")
    # 应用透视变换
    dst_img = cv2.warpPerspective(img, m, img.shape[:2])
    # 显示图像
    cv2.imshow("src_img", img)
    cv2.imshow("dst_img", dst_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 透视变换 detail
def perspective_transformation_detail(img):
    # 定义源和目标图像的4个点，这里至少需要4个点，且点的数量相同
    src_points = np.float32([[50, 50], [200, 50], [200, 200], [50, 200]])
    dst_points = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    # 获取点的数量
    points_numbers = src_points.shape[0]
    print(points_numbers)
    # 通过公式 A * warpMatrix = B , 求解warpMatrix
    A = np.zeros((2*points_numbers, 8))
    B = np.zeros((2*points_numbers, 1))
    for i in range(points_numbers):
        a_i = src_points[i, :]
        b_i = dst_points[i, :]
        A[2*i, :] = [a_i[0], a_i[1], 1, 0, 0, 0, -a_i[0]*b_i[0], -a_i[1]*b_i[0]]
        B[2*i] = b_i[0]
        A[2*i+1, :] = [0, 0, 0, a_i[0], a_i[1], 1, -a_i[0]*b_i[1], -a_i[1]*b_i[1]]
        B[2*i+1] = b_i[1]
    A = np.mat(A)  # 将numpy数组A转换为矩阵, 可以使用运算符 *
    # 注意 warpMatrix 是一个列向量, 数据类型为 numpy.matrix
    warpMatrix = A.I * B  # 这一步可以通过 A * warpMatrix = B  两边同时乘以 A.I得到，因为 A.I * A = 单位矩阵， 单位矩阵 * warp = warp
    # 转为一维行向量
    warpMatrix = np.array(warpMatrix).T[0]  # [0]是为了获取转置后唯一的一行
    # 插入最后一个元素
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    # 转为3*3数组
    warpMatrix = warpMatrix.reshape((3, 3))

    # 进行透视变换
    # dst_img = np.zeros_like(img)
    # for y in range(img.shape[0]):
    #     for x in range(img.shape[1]):
    #         src_point = np.array([x, y, 1])
    #         dst_point = warpMatrix @ src_point  # 矩阵乘法运算符 @
    #         dst_point /= dst_point[2]  # 通过将齐次坐标除以其第三个分量，将其还原为二维坐标，即归一化。
    #         # 确保变换后的目标点坐标 dst_point 在目标图像的范围内
    #         if 0 <= dst_point[0] < img.shape[1] and 0 <= dst_point[1] < img.shape[0]:
    #             if len(img.shape) == 2:
    #                 dst_img[int(dst_point[1]), int(dst_point[0])] = img[y, x]
    #             elif len(img.shape) == 3:
    #                 dst_img[int(dst_point[1]), int(dst_point[0]), :] = img[y, x, :]
    #
    # # 显示图像
    # cv2.imshow("src_img", img)
    # cv2.imshow("dst_img", dst_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    image = cv2.imread("lenna.png")
    perspective_transformation(image)
    perspective_transformation_detail(image)