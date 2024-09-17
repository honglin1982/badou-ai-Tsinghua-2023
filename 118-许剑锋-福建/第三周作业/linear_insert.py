'''
线性插值
'''
import numpy as np
import cv2
from math import floor


def show_img(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()


def near_insert(img, new_width, new_height):
    width_ratio = new_width / img.shape[0]
    height_ratio = new_height / img.shape[1]
    print(width_ratio, height_ratio)
    # 此处uint8和int8效果差别很大：np.int8范围-128-127 np.uint80-255
    # new_img = np.zeros((new_width, new_height), np.uint8)
    new_img = np.zeros((new_width, new_height), np.int8)
    for i in range(new_width):
        for j in range(new_height):
            to_height = int(j / height_ratio + 0.5)
            to_width = int(i / width_ratio + 0.5)
            new_img[i][j] = img[to_width][to_height]
    print(new_img)
    return new_img

'''
双线性插值
'''
def linear_insert(img, new_width, new_height):
    width_ratio = new_width / img.shape[0]
    height_ratio = new_height / img.shape[1]
    new_img = np.zeros((new_width, new_height), np.uint8)
    for i in range(new_width):
        for j in range(new_height):
            y = j / height_ratio
            x = i / width_ratio
            x1 = int(x)
            y1 = int(y)
            x2 = x1 + 1 if x1 < img.shape[0] - 1 else x1
            y2 = y1 + 1 if y1 < img.shape[1] - 1 else y1

            if x1 == x2 and y1 == y2:
                new_img[i][j] = img[x1][y1]
                continue
            if x1 != x2 and y1 != y2:
                # 第一次拟合
                pixel1 = img[x1][y1] + (img[x2][y1] - img[x1][y1]) * (x - x1) / (x2 - x1)

                pixel2 = img[x1][y2] + (img[x2][y2] - img[x1][y2]) * (x - x1) / (x2 - x1)

                # 第二次拟合
                new_img[i][j] = pixel1 + (pixel2 - pixel1) * (y - y1) / (y2 - y1)
            elif x1 == x2:

                new_img[i][j] = img[x1][y1] + (img[x1][y2] - img[x1][y1]) * (y - y1) / (y2 - y1)
            else:
                new_img[i][j] = img[x1][y2] + (img[x2][y2] - img[x1][y2]) * (x - x1) / (x2 - x1)



    return new_img

'''
双线性插值优化
'''
def linear_insert2(img, new_width, new_height):
    width_ratio = new_width / img.shape[0]
    height_ratio = new_height / img.shape[1]
    new_img = np.zeros((new_width, new_height), np.int8)
    for i in range(new_width - 1):
        for j in range(new_height - 1):
            # 平移，使得几何中心位置一致
            y = (j + 0.5) / height_ratio - 0.5
            x = (i + 0.5) / width_ratio - 0.5
            x1 = int(x)
            y1 = int(y)
            x2 = min(x1 + 1, img.shape[0] - 1)
            y2 = min(y1 + 1, img.shape[1] - 1)
            # print(x1, y1, x2, y2)

            # 第一次拟合
            # pixel1 = img[x1][y1] + (img[x2][y1] - img[x1][y1]) * (x - x1) / (x2 - x1)
            pixel1 = (img[x1][y1] * (x2 - x) + (x - x1) * img[x2][y1]) / (x2 - x1)

            # pixel2 = img[x1][y2] + (img[x2][y2] - img[x1][y2]) * (x - x1) / (x2 - x1)
            pixel2 = (img[x1][y2] * (x2 - x) + (x - x1) * img[x2][y2]) / (x2 - x1)
            # 第二次拟合
            pixel = pixel1 + (pixel2 - pixel1) * (y - y1) / (y2 - y1)
            p = int(pixel)
            new_img[i][j] = p
            # new_img[i][j] = int(pixel1 + (pixel2 - pixel1) * (y - y1) / (y2 - y1))
            print('int转换前：{}, 转换后:{}, int转换：{}'.format(pixel, new_img[i][j], p))

    return new_img




if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_img(gray, 'gray')
    print(gray.shape)
    # near_linear = near_insert(gray, 800, 800)
    near_linear = linear_insert2(gray, 1000,1000)
    show_img(near_linear, 'near_linear')




