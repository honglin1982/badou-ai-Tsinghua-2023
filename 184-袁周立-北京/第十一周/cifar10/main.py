import keras
from loader import load_data
from model import CiFar10_Model
import numpy as np


'''
1.实现cifar-10 2.实现alexnet 3.实现vgg 4.实现resnet
'''
def main():
    data_path = "./cifar-10-batches-py"
    train_data, train_label, test_data, test_label = load_data(data_path)
    train_data_num = train_data.shape[0]
    img_shape = train_data[0].shape
    batch_size = 128

    # 数据迭代器
    image_data_gen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.2,
    )
    data_gen = image_data_gen.flow(train_data, train_label, batch_size=batch_size)

    # 模型
    model = CiFar10_Model(img_shape)

    # 学习率下降的方式
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='acc', factor=0.9, patience=3
    )

    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"],
    )

    model.fit_generator(
        data_gen,
        epochs=12,
        steps_per_epoch=train_data_num // batch_size,
        callbacks=[reduce_lr]
    )
    evaluate_result = model.evaluate(test_data, test_label, batch_size=128)
    print("test loss, test acc:", evaluate_result)

    predict = model.predict(test_data)
    correct = int(np.sum(np.argmax(predict, axis=1) == test_label))
    print("correct: %d, acc: %.2f" % (correct, correct / test_label.shape[0]))


if __name__ == "__main__":
    main()