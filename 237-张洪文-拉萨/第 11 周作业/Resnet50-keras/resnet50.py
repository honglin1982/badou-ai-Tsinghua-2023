import numpy as np
from tensorflow.keras import layers, models, preprocessing
import tensorflow.keras.backend as K
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input


# 恒等块: 用于构建深度残差网络；可串联，用于加深网络
def identity_block(input_tensor, kernel_size, filters, stage, block):
    # 确定 3个卷积层的通道数
    filters1, filters2, filters3 = filters

    conv_name_base = f"res{str(stage)}{block}_branch"
    bn_name_base = f"bn{str(stage)}{block}_branch"

    # 分支1
    # 二维卷积层，指定层名，输入数据 input_tensor
    x = layers.Conv2D(filters=filters1, kernel_size=(1,1), name=conv_name_base + "2a")(input_tensor)
    # 批量归一化层，指定层名，(x) 为函数调用标准形式，x 为输入数据
    x = layers.BatchNormalization(name=bn_name_base + "2a")(x)
    x = layers.Activation("relu")(x) # 激活函数：Relu

    # 二维卷积层，输入数据 x, 使用0填充
    x = layers.Conv2D(filters2, kernel_size=kernel_size, padding="same", name=conv_name_base+"2b")(x)
    x = layers.BatchNormalization(name=bn_name_base+"2b")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters3, kernel_size=(1,1), name=conv_name_base+"2c")(x)
    x = layers.BatchNormalization(name=bn_name_base+"2c")(x)

    # 分支2 为空，直接使用原始输入
    # 跳跃连接（残差连接）: 对两个张量执行元素级加法
    x = layers.add([x, input_tensor])
    x = layers.Activation("relu")(x)

    return x

# 残差块：改变网络维度
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
    filters1, filters2, filters3 = filters

    # 定义层名
    conv_name_base = f"res{str(stage)}{block}_branch"
    bn_name_base = f"bn{str(stage)}{block}_branch"

    # 分支1
    # 二维卷积，输入张量 input_tensor，
    x = layers.Conv2D(filters1, kernel_size=(1,1), strides=strides, name=conv_name_base+"2a")(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base+"2a")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters2, kernel_size=kernel_size, padding="same", name=conv_name_base+"2b")(x)
    x = layers.BatchNormalization(name=bn_name_base+"2b")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters3, kernel_size=(1,1), name=conv_name_base+"2c")(x)
    x = layers.BatchNormalization(name=bn_name_base+"2c")(x)

    # 分支2：指定了步幅，使其和分支1的维度相等
    shortcut = layers.Conv2D(filters3, kernel_size=(1,1), strides=strides, name=conv_name_base+"1")(input_tensor)
    shortcut = layers.BatchNormalization(name=bn_name_base+"1")(shortcut)

    # 残差连接
    x = layers.add([x, shortcut])
    x = layers.Activation("relu")(x)

    return x


# 构建 ResNet50 网络
def ResNet50(input_shape=(224,224,3), classes=1000):
    # 输入层：定义输入张量
    img_input = layers.Input(shape=input_shape)
    # 对输入特征图添加 3行3列 的0填充，
    x = layers.ZeroPadding2D(padding=(3,3))(img_input)
    # 卷积、批量归一、激活、最大池化
    x = layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), name="conv1")(x)
    x = layers.BatchNormalization(name="bn_conv1")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    x = conv_block(input_tensor=x, kernel_size=3, filters=[64,64,256], stage=2, block="a", strides=(1,1))
    x = identity_block(x, kernel_size=3, filters=[64,64,256], stage=2, block="b")
    x = identity_block(x, 3, [64,64,256], stage=2, block="c")

    x = conv_block(x, 3, [128,128,512], stage=3, block="a")
    x = identity_block(x, 3, [128,128,512], stage=3, block="b")
    x = identity_block(x, 3, [128,128,512], stage=3, block="c")
    x = identity_block(x, 3, [128,128,512], stage=3, block="d")

    x = conv_block(x, 3, [256,256,1024], stage=4, block="a")
    x = identity_block(x, 3, [256,256,1024], stage=4, block="b")
    x = identity_block(x, 3, [256,256,1024], stage=4, block="c")
    x = identity_block(x, 3, [256,256,1024], stage=4, block="d")
    x = identity_block(x, 3, [256,256,1024], stage=4, block="e")
    x = identity_block(x, 3, [256,256,1024], stage=4, block="f")

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b")
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c")

    # 平均池化
    x = layers.AveragePooling2D((7,7), name="avg_pool")(x)
    x = layers.Flatten()(x)  # 展平为1维张量
    x = layers.Dense(classes, activation="softmax", name="fc1000")(x)  # 全连接层: softmax激活， 类 1000
    # 定义模型，指定输入为输入层的输出， 输出为最后输出层的输出
    model = models.Model(img_input, x, name="resnet50")
    # 加载神经网络模型权重
    model.load_weights(filepath="resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model


if __name__ == '__main__':
    model = ResNet50()  # 获取模型对象
    model.summary()  # 打印模型摘要信息

    # 定义图片路径, 加载图片
    # img_path = "elephant.jpg"
    img_path = "bike.jpg"
    # 加载图像文件并将其转换为 `PIL` 图像对象，大小为 （224，224）
    img = preprocessing.image.load_img(img_path, target_size=(224,224))
    # PIL 对象转 numpy 数组
    x = preprocessing.image.img_to_array(img)
    # 在数组 x 的第一个轴添加一个维度
    x = np.expand_dims(x, axis=0)
    # 使图像数据适用于使用在 ImageNet 上预训练的深度学习模型。
    x = preprocess_input(x)

    print("Input image shape:", x.shape)
    preds = model.predict(x)
    decoded_pred = decode_predictions(preds)
    for i, (imagenet_id, label, score) in enumerate(decoded_pred[0]):
        print(f"{i + 1}: {label} ({score:.9f})")
