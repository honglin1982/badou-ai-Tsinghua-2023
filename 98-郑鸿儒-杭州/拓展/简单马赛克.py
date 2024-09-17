import cv2


def pixelate(src, window=30):
    height, width, channels = src.shape
    img = src
    for channel in range(channels):
        for h in range(height):
            deal_h = h // window * window
            for w in range(width):
                img[hc, wc] = img[deal_h, w // window * window]
    return img


img_src = cv2.imread("lenna.png")
img_pixel = pixelate(img_src)
img_src = cv2.imread("lenna.png")
cv2.imshow("img_src", img_src)
cv2.imshow("pixelate", img_pixel)
cv2.waitKey()
