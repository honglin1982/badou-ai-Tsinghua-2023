import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
import AlexNet_Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 设置图像的维度顺序 channels_first为通道在后(batch_size,h,w,c)  channels_last为(bitch_size,c,h,w)
K.set_image_data_format('channels_last')
def img_resize(imgs, size):
    images = []
    for i in imgs:
        images.append(cv2.resize(i, size))
    images = np.array(images)
    return images


def generate_array_from_file(lines, batch_size):
    total = len(lines)
    i = 0
    while 1:
        x_train = []
        y_train = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            img = cv2.imread("./data/image/train" + '/' + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/255
            x_train.append(img)
            label = lines[i].split(';')[1]
            y_train.append(label)
            # 读完一个周期后重新开始
            i = (i + 1) % total
        x_train = img_resize(x_train, (224, 224))
        x_train = x_train.reshape((-1, 224, 224, 3))
        y_train = to_categorical(np.array(y_train), num_classes=2)
        yield (x_train, y_train)





if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    with open(r".\data\dataset.txt","r") as f:
        lines = f.readlines()

    # 随机打乱
    np.random.shuffle(lines)

    #分为两组 一组训练 一组测试
    total = len(lines)
    train_num = int(total * 0.9)
    val_num = total - train_num

    # 保存的方式，3世代保存一次 save_weights_only只保存权重，false可以保存模型结构等
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=3
    )
    # 学习率下降的方式，acc  patience = 三次不下降就下降学习率继续训练 lr *= 0.5(factor)
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    network = AlexNet_Model.model()
    # 交叉熵
    network.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    batch_size = 128
    # fit_generator不用设置batch_size 所以需要设置迭代次数 iterate
    network.fit_generator(generate_array_from_file(lines[:train_num], batch_size),
                          steps_per_epoch=max(1, train_num // batch_size),
                          validation_data=generate_array_from_file(lines[train_num:], batch_size),
                          validation_steps=max(1, val_num // batch_size),
                          epochs=50,
                          initial_epoch=0,
                          callbacks=[checkpoint_period1, reduce_lr])
    network.save_weights(log_dir + 'alexnet_model.h5')
    print("#######   FINISHED   ########")







