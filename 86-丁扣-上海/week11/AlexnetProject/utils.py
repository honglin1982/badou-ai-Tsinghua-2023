import tensorflow as tf
import numpy as np
import cv2


with open(f'./data/classify.txt') as fp:
    lines = fp.readlines()
    print(lines)
    result_map = [str(l).split(';')[1][:1] for l in lines]


def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for img in image:
            _img = cv2.resize(img, size)
            images.append(_img)
        images = np.array(images)
        return images


def output_answer(i: int):
    print(result_map)
    return result_map[i]


if __name__ == '__main__':
    output_answer(0)
    pass

