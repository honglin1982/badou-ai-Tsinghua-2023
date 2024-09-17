import keras
import cv2
import numpy as np
from model import AlexNet_Model
from diy_generator import DiyGenerator



def main():

    batch_size = 128
    total_data_num = 25000
    valid_rate = 0.1

    num_train = int(total_data_num * (1 - valid_rate))
    num_test = total_data_num - num_train

    name_format = lambda index: "{}.{}.jpg".format("dog", index - 12500) if index > 12499 \
        else "{}.{}.jpg".format("cat", index)
    data_gen = DiyGenerator("./image/train", name_format, total_data_num,
                            valid_rate=valid_rate,
                            batch_size=batch_size,
                            target_size=(227, 227))
    train_data_gen, test_data_gen = data_gen.get_train_gen(), data_gen.get_test_gen()

    # 3epoch保存一次模型
    log_dir = "./logs/"
    checkpoint = keras.callbacks.ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        period=3
    )

    # 早停
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5
    )

    # 构造模型
    model = AlexNet_Model(output_shape=2)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.0001),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"]
    )

    # 拟合数据
    model.fit_generator(
        train_data_gen,
        steps_per_epoch=num_train//batch_size,
        validation_data=test_data_gen,
        validation_steps=num_test//batch_size,
        epochs=12,
        callbacks=[checkpoint, early_stop]
    )

    # 保存模型
    model.save_weights(log_dir + 'last.h5')


if __name__ == "__main__":
    main()
