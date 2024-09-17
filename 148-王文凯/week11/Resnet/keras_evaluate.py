import numpy as np
import os
import cv2
from keras_network import resnet_50

from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def main():
    images = []
    for root, sub_folders, files in os.walk('./images'):
        for file in files:
            images.append(
                cv2.resize(
                    cv2.cvtColor(cv2.imread(os.path.join(root, file)), cv2.COLOR_BGR2RGB),
                    (224, 224),
                    interpolation=cv2.INTER_LINEAR)
            )
    images = np.array(images)
    x = preprocess_input(images)

    model = resnet_50()
    predictions = decode_predictions(model.predict(x))
    for prediction in predictions:
        print('predictions:', prediction)


if __name__ == '__main__':
    main()
