import cv2


# 差值hash
def difference_hash(img, width=9, height=8):
    # 图像缩放: size=(8,9) 8行9列
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    # 转为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray_img)
    # 比较生成hash
    img_hash = ""
    for i in range(8):  # 8行
        for j in range(8):  # 9列，8个差值
            if gray_img[i, j] > gray_img[i, j+1]:
                img_hash += "1"
            else:
                img_hash += "0"
    print(img_hash)
    return img_hash


if __name__ == '__main__':
    image = cv2.imread("image/lenna.png")
    difference_hash(image)
