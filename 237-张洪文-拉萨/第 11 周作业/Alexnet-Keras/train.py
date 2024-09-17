import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import utils  # 自定义的工具模块
from model.AlexNet import AlexNet  # 自定义的网络模型
from tensorflow.keras import backend as K  # 后台引擎设置
K.set_image_data_format("channels_last")  # 设置图像数据的维度顺序,顺序=（h,w,c）

# 读取文件生成数组
def generate_arrays_from_file(lines, batch_size):
    n = len(lines)
    i = 0
    while True:
        X_train = []
        Y_train = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(";")[0]  # 获取文件名
            y = lines[i].split(";")[1]  # 标签值
            # 读取图像
            img = cv2.imread(r"./data/image/train/" + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/255
            X_train.append(img)
            Y_train.append(y)
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = utils.resize_image(X_train, (224,224))
        X_train = X_train.reshape(-1,224,224,3)  # -1 自动计算维度
        Y_train = to_categorical(np.array(Y_train), num_classes=2)  # 分2类
        yield X_train, Y_train


if __name__ == '__main__':
    # 设置模型文件名
    log_dir = "./logs/"  # 模型保存目录
    # 模型名：当前训练周期数、训练集上的损失值和验证集上的损失值
    log_filename = "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"

    # 读取数据集，数据集格式：图片名;类别索引
    with open("./data/dataset.txt", "r") as f:
        lines = f.readlines()
    # 进行随机初始化
    np.random.seed(10101)  # 设置随机数种子
    np.random.shuffle(lines)  # 打乱数据集的顺序
    np.random.seed(None)  # 恢复默认值
    # 设置训练样本数=90%，估计样本数=10%
    number_val = int(len(lines) * 0.1)
    number_train = len(lines) - number_val

    # 建立AlexNet模型
    model = AlexNet()
    # 创建 ModelCheckpoint 回调函数:保存模型的权重和架构
    checkpoint = ModelCheckpoint(
        filepath=log_dir + log_filename,
        monitor="acc",  # 监测指标
        save_weights_only=False,  # 不仅仅是只保存权重，还要保存架构
        save_best_only=True,  # 仅在监测指标有所改善时才保存模型
        period=5  # 每5个周期保存一次
    )
    # 创建 ReduceLROnPlateau 回调函数:在训练期间根据某个指标的表现自动降低学习率
    reduce_lr = ReduceLROnPlateau(
        monitor="acc",  # 监测指标
        factor=0.5,  # 学习率因子
        patience=3,  # 在监测指标停止改善的情况下，等待3个训练周期来降低学习率
        verbose=1    # 输出学习率降低信息
    )
    # 创建 EarlyStopping 回调函数:避免过拟合
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0,  # 新的指标值与前一个值的差异小于 `min_delta`，则被认为没有改善
        patience=10,  # 当 val_loss 10个训练周期不下降时,提前停止训练
        verbose=1
    )
    # 定义模型的优化器、损失函数以及评估指标
    model.compile(
        loss="categorical_crossentropy",  # 分类问题选择的损失函数
        optimizer=Adam(lr=1e-3),  # 设置学习率 0.001
        metrics=["accuracy"],  # 评估指标
    )

    batch_size = 128  # 一次的训练集大小
    print("Train on {} samples, val on {} samples, with bath size {}.".format(number_train, number_val, batch_size))

    # 开始训练
    model.fit_generator(
        # 数据生成器
        generator=generate_arrays_from_file(lines[:number_train], batch_size),
        steps_per_epoch=max(1, number_train//batch_size),  # 每个训练周期（epoch）的步数
        validation_data=generate_arrays_from_file(lines[number_train:], batch_size),
        validation_steps=max(1, number_val//batch_size),
        epochs=50,
        initial_epoch=0,
        callbacks=[checkpoint, reduce_lr, early_stopping]
    )
    model.save_weights(log_dir + "last1.h5")
