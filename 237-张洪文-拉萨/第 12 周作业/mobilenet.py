import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import backend as K


# 使用后端的relu函数，并设置最大值为 6，防止激活值爆炸
def relu6(x):
    x = K.relu(x, max_value=6)
    return x

# 深度可分离卷积层:
def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1,1), block_id=1):
    # 深度可分离卷积
    x = layers.DepthwiseConv2D(kernel_size=(3,3), padding="same",
                               depth_multiplier=depth_multiplier,
                               strides=strides, use_bias=False,
                               name="conv_dw_%d" % block_id)(inputs)
    x = layers.BatchNormalization(name="conv_dw_%d_bn" % block_id)(x)
    x = layers.Activation(activation=relu6, name="conv_dw_%d_relu" % block_id)(x)

    x = layers.Conv2D(filters=pointwise_conv_filters, kernel_size=(1,1),
                      padding="same", use_bias=False, strides=(1,1),
                      name="conv_pw_%d" % block_id)(x)
    x = layers.BatchNormalization(name="conv_pw_%d_bn" % block_id)(x)
    x = layers.Activation(activation=relu6, name="conv_pw_%d_relu" % block_id)(x)
    return x


# 网络部分
def MobileNet(
        input_shape=(224,224,3), depth_multiplier=1, dropout=1e-3, classes=1000):

    # 指定输入图像 shape
    img_input = layers.Input(shape=input_shape)

    # 224x224x3 -> 112x112x32
    x = layers.Conv2D(filters=32, kernel_size=(3,3), padding="same",
                      use_bias=False, strides=(2,2), name="conv1")(img_input)
    x = layers.BatchNormalization(name="conv1_bn")(x)
    x = layers.Activation(activation=relu6, name="conv1_relu")(x)

    # 112x112x32 -> 112x112x64
    x = _depthwise_conv_block(x, 64, depth_multiplier=depth_multiplier, block_id=1)

    # 112x112x64 -> 56x56x128
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2,2), block_id=2)

    # 56x56x128 -> 56x56x128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 56x56x128 -> 28x28x256
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2,2), block_id=4)

    # 28x28x256 -> 28x28x256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 28x28x256 -> 14x14x512
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(2,2), block_id=6)

    # 14x14x512 -> 14x14x512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14x14x512 -> 7x7x1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(2,2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 7x7x1023 -> 1x1x1024
    x = layers.GlobalAveragePooling2D()(x)  # 全局平均池化
    x = layers.Reshape(target_shape=(1,1,1024), name="reshape_1")(x)  # 用于改变形状
    x = layers.Dropout(rate=dropout, name="dropout")(x)  # 丢弃层
    x = layers.Conv2D(classes, kernel_size=(1,1), padding="same", name="conv_preds")(x)
    x = layers.Activation(activation="softmax", name="activation_softmax")(x)
    x = layers.Reshape(target_shape=(classes,), name="reshape_2")(x)

    inputs = img_input
    model = Model(inputs, x, name="mobilenet_1_0_224_tf")

    # 加载模块
    model_name = "mobilenet_1_0_224_tf.h5"
    model.load_weights(filepath=model_name)

    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = MobileNet(input_shape=(224,224,3))

    img_path = "elephant.jpg"
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print("Input image shape: ", x.shape)

    predicts = model.predict(x)
    print(np.argmax(predicts))  # 最大值索引
    print("Prediction result Top3:", decode_predictions(predicts, 3))
