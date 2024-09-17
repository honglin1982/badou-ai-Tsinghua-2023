import cv2
import numpy as np
from numpy import ndarray


def nearest_interpolation(image_file: ndarray, target_height: int, target_width: int):
    source_height, source_width, channels = image_file.shape
    empty_image = np.zeros((target_height, target_width, channels), np.uint8)
    for target_x in range(target_width):
        for target_y in range(target_height):
            virtual_x = target_x * (source_width / target_width)
            virtual_y = target_y * (source_height / target_height)
            nearest_source_x = round(virtual_x)
            nearest_source_y = round(virtual_y)
            empty_image[target_y, target_x] = image_file[nearest_source_y, nearest_source_x]

    return empty_image


if __name__ == '__main__':
    image = cv2.imread("lenna.png")
    zoomed_image = nearest_interpolation(image_file=image, target_height=700, target_width=700)
    print(zoomed_image)
    print(zoomed_image.shape)
    cv2.imshow("nearest interpolation", zoomed_image)
    cv2.imshow("source image", image)
    cv2.waitKey(0)
