# 实现二值化
import cv2 as cv


def threshold_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("灰度图像", gray)
    # 最大类间方差法,第2参数无意义，因为是全局自适应阈值，常用于双峰直方图
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("阈值：%s" % ret)
    cv.imshow("最大类间方差法", binary)
    # 三角法TRIANGLE全局自适应阈值类型,用于单峰直方图
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    print("阈值：%s" % ret)
    cv.imshow("三角法TRIANGLE", binary)
    # 自定义阈值为150,大于150的是白色 小于的是黑色
    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
    print("阈值：%s" % ret)
    cv.imshow("自定义", binary)
    # 自定义阈值为150,大于150的是黑色 小于的是白色
    ret, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
    print("阈值：%s" % ret)
    cv.imshow("自定义反色", binary)


src = cv.imread("7.jpg")
threshold_image(src)
cv.waitKey(0)
cv.destroyAllWindows()


