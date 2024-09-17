from keras.datasets import cifar100
import cv2
import numpy as np


def get_data(tag='train', img_shape=(224, 224), size=5000, label_process=True):
    (images, labels), _ = cifar100.load_data()
    if tag == 'evaluate':
        images, labels = _
    images = image_resize(images, img_shape, num_image=size)
    if label_process:
        labels = [item[0] for item in labels[:size]]
    else:
        labels = labels[:size]
    return images, labels


def image_resize(images, size, num_image=50000, normalize=True):
    new_images = []
    for image in images[:num_image]:
        image = cv2.resize(image, size)
        image = image.astype('float32')
        new_images.append(image)
    new_images = np.array(new_images)
    if normalize:
        new_images /= 255
    return new_images
