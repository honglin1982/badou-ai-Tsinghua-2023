
import numpy as np
import random
import cv2


def show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

def gaussion_noise(img, mean, sigma):
    '''
    高斯噪声
    :param img:
    :param mean:
    :param sigma:
    :return:
    '''
    w = img.shape[0]
    h = img.shape[1]
    new_img = np.zeros((w, h), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            pixel = img[i][j] + random.gauss(mean, sigma)
            if pixel > 255:
                pixel = 255
            if pixel < 0:
                pixel = 0

            new_img[i][j] = pixel

    return new_img

def random_gaussion_noise(img, mean, sigma, percentage):
    '''
    高斯噪声
    :param img:
    :param mean:
    :param sigma:
    :return:
    '''
    w = img.shape[0]
    h = img.shape[1]
    new_img = img
    random_cnt = int(w * h * percentage)
    for i in range(random_cnt):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        pixel = new_img[x][y] + random.gauss(mean, sigma)
        if pixel > 255:
            pixel = 255
        if pixel < 0:
            pixel = 0

        new_img[x][y] = pixel

    return new_img


def salt_pepper_noise(img, percentage):
    '''
    椒盐噪声
    :param img:
    :param percentage:
    :return:
    '''
    new_img = img.copy() # 此处需要深度拷贝，否则会修改原图，影响后续调用的结果
    w = img.shape[0]
    h = img.shape[1]
    noise_cnt = int(w * h * percentage)
    for i in range(noise_cnt):
        print(i)
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        ran = np.random.random()
        if ran >= 0.5:
            new_img[x][y] = 255
        else:
            new_img[x][y] = 0
    return new_img



if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gauss_img = gaussion_noise(gray, 2.4, 0.8)
    # random_guass_img = random_gaussion_noise(gray, 2.4, 0.8, 0.8)
    #
    show(gray, 'gray')
    # show(gauss_img, 'gauss_img')
    # show(random_guass_img, 'random_guass_img')
    salt_2 = salt_pepper_noise(gray, 0.001)
    salt_8 = salt_pepper_noise(gray, 0.00001)
    show(salt_2, 'salt_2')
    show(salt_8, 'salt_8')
    show(gray, 'gray')



