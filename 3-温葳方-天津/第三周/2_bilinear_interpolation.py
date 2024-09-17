import cv2
import numpy as np
from numpy import ndarray


def bilinear_interpolation(image_file: ndarray, target_height: int, target_width: int):
    source_height, source_width, channels = image_file.shape

    # Return source image if target image is at the same size of source image.
    if source_height == target_height and source_width == target_width:
        return image_file.copy()

    target_image = np.zeros((target_height, target_width, channels), dtype=np.uint8)
    for channel in range(channels):
        for target_x in range(target_width):
            for target_y in range(target_height):
                # To move the center of the two images into the same spot,
                # We have a function: srcX + 0.5 = (dstX + 0.5) ∗ (srcWidth / dstWidth)
                virtual_x = (target_x + 0.5) * (source_width / target_width) - 0.5
                virtual_y = (target_y + 0.5) * (source_height / target_height) - 0.5

                small_source_x = int(np.floor(virtual_x))
                big_source_x = min(small_source_x + 1, source_width - 1)
                small_source_y = int(np.floor(virtual_y))
                big_source_y = min(small_source_y + 1, source_height - 1)

                # Bilinear interpolation formula. It's in PPT.
                # Attention that x2-x1=1, y2-y1=1, (big_source_x-small_source_x=1,) So these two parts are omitted.
                function_value_r1 = (big_source_x - virtual_x) * image_file[small_source_y, small_source_x, channel] + (
                        virtual_x - small_source_x) * image_file[small_source_y, big_source_x, channel]
                function_value_r2 = (big_source_x - virtual_x) * image_file[big_source_y, small_source_x, channel] + (
                        virtual_x - small_source_x) * image_file[big_source_y, big_source_x, channel]
                target_image[target_y, target_x, channel] = (big_source_y - virtual_y) * function_value_r1 + (
                        virtual_y - small_source_y) * function_value_r2
    return target_image


if __name__ == '__main__':
    image = cv2.imread("lenna.png")
    zoomed_image = bilinear_interpolation(image_file=image, target_height=700, target_width=700)
    print(zoomed_image)
    print(zoomed_image.shape)
    cv2.imshow("bilinear interpolation", zoomed_image)
    cv2.imshow("source image", image)
    cv2.waitKey(0)
