import tensorflow as tf
import time


class InceptionV3:
    def __init__(self, input_shape=[229, 229, 3], output_shape=1000):
        self.x = tf.placeholder(tf.float32, [None] + input_shape)
        self.y = tf.placeholder(tf.int32, [None, output_shape])
        self.is_training = tf.placeholder(tf.bool)

        # 299 * 299 * 3 -> 149 * 149 * 32
        conv_1 = self.conv2d_bn(
            inputs=self.x,
            filters=32,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(2, 2),
            padding='VALID',
            name='conv_1'
        )

        # 149 * 149 * 32 -> 147 * 147 * 32
        conv_2 = self.conv2d_bn(
            inputs=conv_1,
            filters=32,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(1, 1),
            padding='VALID',
            name='conv_2'
        )

        # 147 * 147 * 32 -> 147 * 147 * 64
        conv_3 = self.conv2d_bn(
            inputs=conv_2,
            filters=64,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(1, 1),
            padding='SAME',
            name='conv_3'
        )

        # 147 * 147 * 64 -> 73 * 73 * 64
        pool_1 = tf.layers.max_pooling2d(
            inputs=conv_3,
            pool_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            name='pool_1'
        )

        # 73 * 73 * 64 -> 71 * 71 * 80
        conv_4 = self.conv2d_bn(
            inputs=pool_1,
            filters=80,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(1, 1),
            padding='VALID',
            name='conv_4'
        )

        # 71 * 71 * 80 -> 35 * 35 * 192
        conv_5 = self.conv2d_bn(
            inputs=conv_4,
            filters=192,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(2, 2),
            padding='VALID',
            name='conv_5'
        )

        # 35 * 35 * 192 -> 35 * 35 * 288
        conv_6 = self.conv2d_bn(
            inputs=conv_5,
            filters=288,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(1, 1),
            padding='SAME',
            name='conv_6'
        )

        # inception module 1 part 1
        # 35 * 35 * 288 -> 17 * 17 * 512
        # 512 = 128 + 128 + 128 + 128
        branch1x1_1_1 = self.conv2d_bn(
            inputs=conv_6,
            filters=128,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(2, 2),
            padding="VALID",
            name="inception_module_1_1_branch_1x1"
        )

        branch5x5_1_1 = self.conv2d_bn(
            inputs=conv_6,
            filters=96,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_1_branch_1x1"
        )
        branch5x5_1_1 = self.conv2d_bn(
            inputs=branch5x5_1_1,
            filters=128,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_1_branch_5x5"
        )
        branch5x5_1_1 = self.conv2d_bn(
            inputs=branch5x5_1_1,
            filters=128,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(2, 2),
            padding="VALID",
            name="inception_module_1_1_branch_5x5"
        )

        branch3x3_1_1 = self.conv2d_bn(
            inputs=conv_6,
            filters=96,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_1_branch_3x3"
        )
        branch3x3_1_1 = self.conv2d_bn(
            inputs=branch3x3_1_1,
            filters=128,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_1_branch_3x3"
        )
        branch3x3_1_1 = self.conv2d_bn(
            inputs=branch3x3_1_1,
            filters=128,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(2, 2),
            padding="VALID",
            name="inception_module_1_1_branch_3x3"
        )

        branch_pool_1_1 = tf.layers.average_pooling2d(
            inputs=conv_6,
            pool_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            name="inception_module_1_1_branch_pool"
        )
        branch_pool_1_1 = self.conv2d_bn(
            inputs=branch_pool_1_1,
            filters=128,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(2, 2),
            padding="VALID",
            name="inception_module_1_1_branch_pool"
        )

        outputs_1_1 = tf.concat(
            [branch1x1_1_1, branch5x5_1_1, branch3x3_1_1, branch_pool_1_1],
            axis=3,
            name='outputs_1_1'
        )

        # inception module 1 part 2
        # 17 * 17 * 512 -> 17 * 17 * 768
        # 768 = 192 + 192 + 192 + 192
        branch1x1_1_2 = self.conv2d_bn(
            inputs=outputs_1_1,
            filters=192,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_2_branch_1x1"
        )

        branch5x5_1_2 = self.conv2d_bn(
            inputs=outputs_1_1,
            filters=160,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_2_branch_1x1"
        )
        branch5x5_1_2 = self.conv2d_bn(
            inputs=branch5x5_1_2,
            filters=192,
            kernel_size=(5, 5),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_2_branch_5x5"
        )

        branch3x3_1_2 = self.conv2d_bn(
            inputs=outputs_1_1,
            filters=160,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_2_branch_3x3"
        )
        branch3x3_1_2 = self.conv2d_bn(
            inputs=branch3x3_1_2,
            filters=192,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_2_branch_3x3"
        )
        branch3x3_1_2 = self.conv2d_bn(
            inputs=branch3x3_1_2,
            filters=192,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_2_branch_3x3"
        )

        branch_pool_1_2 = tf.layers.average_pooling2d(
            inputs=outputs_1_1,
            pool_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_2_branch_pool"
        )
        branch_pool_1_2 = self.conv2d_bn(
            inputs=branch_pool_1_2,
            filters=192,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_2_branch_pool"
        )

        outputs_1_2 = tf.concat(
            [branch1x1_1_2, branch5x5_1_2, branch3x3_1_2, branch_pool_1_2],
            axis=3,
            name='outputs_1_2'
        )

        # inception module 1 part 3
        # 17 * 17 * 768 -> 17 * 17 * 768
        # 768 = 192 + 192 + 192 + 192
        branch1x1_1_3 = self.conv2d_bn(
            inputs=outputs_1_2,
            filters=192,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_3_branch_1x1"
        )

        branch5x5_1_3 = self.conv2d_bn(
            inputs=outputs_1_2,
            filters=160,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_3_branch_1x1"
        )
        branch5x5_1_3 = self.conv2d_bn(
            inputs=branch5x5_1_3,
            filters=192,
            kernel_size=(5, 5),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_3_branch_5x5"
        )

        branch3x3_1_3 = self.conv2d_bn(
            inputs=outputs_1_2,
            filters=160,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_3_branch_3x3"
        )
        branch3x3_1_3 = self.conv2d_bn(
            inputs=branch3x3_1_3,
            filters=192,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_3_branch_3x3"
        )
        branch3x3_1_3 = self.conv2d_bn(
            inputs=branch3x3_1_3,
            filters=192,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_3_branch_3x3"
        )

        branch_pool_1_3 = tf.layers.average_pooling2d(
            inputs=outputs_1_2,
            pool_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_3_branch_pool"
        )
        branch_pool_1_3 = self.conv2d_bn(
            inputs=branch_pool_1_3,
            filters=192,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_1_3_branch_pool"
        )

        outputs_1_3 = tf.concat(
            [branch1x1_1_3, branch5x5_1_3, branch3x3_1_3, branch_pool_1_3],
            axis=3,
            name='outputs_1_3'
        )

        # inception module 2 part 1
        # 17 * 17 * 768 -> 8 * 8 * 1024
        # 1024 = 384 + 384 + 256
        branch3x3_2_1 = self.conv2d_bn(
            inputs=outputs_1_3,
            filters=384,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(2, 2),
            padding="VALID",
            name="inception_module_2_1_branch_1x1"
        )

        branch3x3dbl_2_1 = self.conv2d_bn(
            inputs=outputs_1_3,
            filters=352,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_1_branch_3x3dbl"
        )
        branch3x3dbl_2_1 = self.conv2d_bn(
            inputs=branch3x3dbl_2_1,
            filters=384,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_1_branch_3x3dbl"
        )
        branch3x3dbl_2_1 = self.conv2d_bn(
            inputs=branch3x3dbl_2_1,
            filters=384,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(2, 2),
            padding="VALID",
            name="inception_module_2_1_branch_3x3dbl"
        )
        branch_pool_2_1 = tf.layers.max_pooling2d(
            inputs=outputs_1_3,
            pool_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            name="inception_module_2_1_branch_pool"
        )
        branch_pool_2_1 = self.conv2d_bn(
            inputs=branch_pool_2_1,
            filters=256,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(2, 2),
            padding="VALID",
            name="inception_module_2_1_branch_pool"
        )

        outputs_2_1 = tf.concat(
            [branch3x3_2_1, branch3x3dbl_2_1, branch_pool_2_1],
            axis=3,
            name='outputs_2_1'
        )

        # inception module 2 part 2
        # 8 * 8 * 1024 -> 8 * 8 * 1280
        # 1280 = 320 + 320 + 320 + 320
        branch1x1_2_2 = self.conv2d_bn(
            inputs=outputs_2_1,
            filters=320,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_2_branch_1x1"
        )

        branch7x7_2_2 = self.conv2d_bn(
            inputs=outputs_2_1,
            filters=256,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_2_branch_7x7"
        )
        branch7x7_2_2 = self.conv2d_bn(
            inputs=branch7x7_2_2,
            filters=320,
            kernel_size=(1, 7),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_2_branch_7x7"
        )
        branch7x7_2_2 = self.conv2d_bn(
            inputs=branch7x7_2_2,
            filters=320,
            kernel_size=(7, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_2_branch_7x7"
        )

        branch7x7dbl_2_2 = self.conv2d_bn(
            inputs=outputs_2_1,
            filters=256,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_2_branch_7x7dbl"
        )
        branch7x7dbl_2_2 = self.conv2d_bn(
            inputs=branch7x7dbl_2_2,
            filters=256,
            kernel_size=(7, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_2_branch_7x7dbl"
        )
        branch7x7dbl_2_2 = self.conv2d_bn(
            inputs=branch7x7dbl_2_2,
            filters=256,
            kernel_size=(1, 7),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_2_branch_7x7dbl"
        )
        branch7x7dbl_2_2 = self.conv2d_bn(
            inputs=branch7x7dbl_2_2,
            filters=256,
            kernel_size=(7, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_2_branch_7x7dbl"
        )
        branch7x7dbl_2_2 = self.conv2d_bn(
            inputs=branch7x7dbl_2_2,
            filters=320,
            kernel_size=(1, 7),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_2_branch_7x7dbl"
        )

        branch_pool_2_2 = tf.layers.max_pooling2d(
            inputs=outputs_2_1,
            pool_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_2_branch_pool"
        )
        branch_pool_2_2 = self.conv2d_bn(
            inputs=branch_pool_2_2,
            filters=320,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_2_branch_pool"
        )

        outputs_2_2 = tf.concat(
            [branch1x1_2_2, branch7x7_2_2, branch7x7dbl_2_2, branch_pool_2_2],
            axis=3,
            name='outputs_2_2'
        )

        # inception module 2 part 3
        # 8 * 8 * 1280 -> 8 * 8 * 1280
        # 1280 = 320 + 320 + 320 + 320
        branch1x1_2_3 = self.conv2d_bn(
            inputs=outputs_2_2,
            filters=320,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_3_branch_1x1"
        )

        branch7x7_2_3 = self.conv2d_bn(
            inputs=outputs_2_2,
            filters=256,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_3_branch_7x7"
        )
        branch7x7_2_3 = self.conv2d_bn(
            inputs=branch7x7_2_3,
            filters=320,
            kernel_size=(1, 7),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_3_branch_7x7"
        )
        branch7x7_2_3 = self.conv2d_bn(
            inputs=branch7x7_2_3,
            filters=320,
            kernel_size=(7, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_3_branch_7x7"
        )

        branch7x7dbl_2_3 = self.conv2d_bn(
            inputs=outputs_2_2,
            filters=256,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_3_branch_7x7dbl"
        )
        branch7x7dbl_2_3 = self.conv2d_bn(
            inputs=branch7x7dbl_2_3,
            filters=256,
            kernel_size=(7, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_3_branch_7x7dbl"
        )
        branch7x7dbl_2_3 = self.conv2d_bn(
            inputs=branch7x7dbl_2_3,
            filters=256,
            kernel_size=(1, 7),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_3_branch_7x7dbl"
        )
        branch7x7dbl_2_3 = self.conv2d_bn(
            inputs=branch7x7dbl_2_3,
            filters=256,
            kernel_size=(7, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_3_branch_7x7dbl"
        )
        branch7x7dbl_2_3 = self.conv2d_bn(
            inputs=branch7x7dbl_2_3,
            filters=320,
            kernel_size=(1, 7),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_3_branch_7x7dbl"
        )

        branch_pool_2_3 = tf.layers.max_pooling2d(
            inputs=outputs_2_2,
            pool_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_3_branch_pool"
        )
        branch_pool_2_3 = self.conv2d_bn(
            inputs=branch_pool_2_3,
            filters=320,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_3_branch_pool"
        )

        outputs_2_3 = tf.concat(
            [branch1x1_2_3, branch7x7_2_3, branch7x7dbl_2_3, branch_pool_2_3],
            axis=3,
            name='outputs_2_3'
        )

        # inception module 2 part 4
        # 8 * 8 * 1280 -> 8 * 8 * 1280
        # 1280 = 320 + 320 + 320 + 320
        branch1x1_2_4 = self.conv2d_bn(
            inputs=outputs_2_3,
            filters=320,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_4_branch_1x1"
        )

        branch7x7_2_4 = self.conv2d_bn(
            inputs=outputs_2_3,
            filters=256,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_4_branch_7x7"
        )
        branch7x7_2_4 = self.conv2d_bn(
            inputs=branch7x7_2_4,
            filters=320,
            kernel_size=(1, 7),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_4_branch_7x7"
        )
        branch7x7_2_4 = self.conv2d_bn(
            inputs=branch7x7_2_4,
            filters=320,
            kernel_size=(7, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_4_branch_7x7"
        )

        branch7x7dbl_2_4 = self.conv2d_bn(
            inputs=outputs_2_3,
            filters=256,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_4_branch_7x7dbl"
        )
        branch7x7dbl_2_4 = self.conv2d_bn(
            inputs=branch7x7dbl_2_4,
            filters=256,
            kernel_size=(7, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_4_branch_7x7dbl"
        )
        branch7x7dbl_2_4 = self.conv2d_bn(
            inputs=branch7x7dbl_2_4,
            filters=256,
            kernel_size=(1, 7),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_4_branch_7x7dbl"
        )
        branch7x7dbl_2_4 = self.conv2d_bn(
            inputs=branch7x7dbl_2_4,
            filters=256,
            kernel_size=(7, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_4_branch_7x7dbl"
        )
        branch7x7dbl_2_4 = self.conv2d_bn(
            inputs=branch7x7dbl_2_4,
            filters=320,
            kernel_size=(1, 7),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_4_branch_7x7dbl"
        )

        branch_pool_2_4 = tf.layers.max_pooling2d(
            inputs=outputs_2_3,
            pool_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_4_branch_pool"
        )
        branch_pool_2_4 = self.conv2d_bn(
            inputs=branch_pool_2_4,
            filters=320,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_4_branch_pool"
        )

        outputs_2_4 = tf.concat(
            [branch1x1_2_4, branch7x7_2_4, branch7x7dbl_2_4, branch_pool_2_4],
            axis=3,
            name='outputs_2_3'
        )

        # inception module 2 part 5
        # 8 * 8 * 1280 -> 8 * 8 * 1280
        # 1280 = 320 + 320 + 320 + 320
        branch1x1_2_5 = self.conv2d_bn(
            inputs=outputs_2_4,
            filters=320,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_5_branch_1x1"
        )

        branch7x7_2_5 = self.conv2d_bn(
            inputs=outputs_2_4,
            filters=256,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_5_branch_7x7"
        )
        branch7x7_2_5 = self.conv2d_bn(
            inputs=branch7x7_2_5,
            filters=320,
            kernel_size=(1, 7),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_5_branch_7x7"
        )
        branch7x7_2_5 = self.conv2d_bn(
            inputs=branch7x7_2_5,
            filters=320,
            kernel_size=(7, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_5_branch_7x7"
        )

        branch7x7dbl_2_5 = self.conv2d_bn(
            inputs=outputs_2_4,
            filters=256,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_5_branch_7x7dbl"
        )
        branch7x7dbl_2_5 = self.conv2d_bn(
            inputs=branch7x7dbl_2_5,
            filters=256,
            kernel_size=(7, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_5_branch_7x7dbl"
        )
        branch7x7dbl_2_5 = self.conv2d_bn(
            inputs=branch7x7dbl_2_5,
            filters=256,
            kernel_size=(1, 7),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_5_branch_7x7dbl"
        )
        branch7x7dbl_2_5 = self.conv2d_bn(
            inputs=branch7x7dbl_2_5,
            filters=256,
            kernel_size=(7, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_5_branch_7x7dbl"
        )
        branch7x7dbl_2_5 = self.conv2d_bn(
            inputs=branch7x7dbl_2_5,
            filters=320,
            kernel_size=(1, 7),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_5_branch_7x7dbl"
        )

        branch_pool_2_5 = tf.layers.max_pooling2d(
            inputs=outputs_2_4,
            pool_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_5_branch_pool"
        )
        branch_pool_2_5 = self.conv2d_bn(
            inputs=branch_pool_2_5,
            filters=320,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_2_5_branch_pool"
        )

        outputs_2_5 = tf.concat(
            [branch1x1_2_5, branch7x7_2_5, branch7x7dbl_2_5, branch_pool_2_5],
            axis=3,
            name='outputs_2_5'
        )

        # inception module 3 part 1
        # 8 * 8 * 1280 -> 8 * 8 * 2048
        # 2048 = 784 + 784 + 512
        branch3x3_3_1 = self.conv2d_bn(
            inputs=outputs_2_5,
            filters=512,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_1_branch_3x3"
        )
        branch3x3_3_1 = self.conv2d_bn(
            inputs=branch3x3_3_1,
            filters=784,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_1_branch_3x3"
        )

        branch7x7x3_3_1 = self.conv2d_bn(
            inputs=outputs_2_5,
            filters=784,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_1_branch_7x7x3"
        )
        branch7x7x3_3_1 = self.conv2d_bn(
            inputs=branch7x7x3_3_1,
            filters=784,
            kernel_size=(1, 7),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_1_branch_7x7x3"
        )
        branch7x7x3_3_1 = self.conv2d_bn(
            inputs=branch7x7x3_3_1,
            filters=784,
            kernel_size=(7, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_1_branch_7x7x3"
        )
        branch7x7x3_3_1 = self.conv2d_bn(
            inputs=branch7x7x3_3_1,
            filters=784,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_1_branch_7x7x3"
        )

        branch_pool_3_1 = tf.layers.max_pooling2d(
            inputs=outputs_2_5,
            pool_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_1_branch_pool"
        )
        branch_pool_3_1 = self.conv2d_bn(
            inputs=branch_pool_3_1,
            filters=512,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_1_branch_pool"
        )

        outputs_3_1 = tf.concat(
            [branch3x3_3_1, branch7x7x3_3_1, branch_pool_3_1],
            axis=3,
            name='outputs_3_1'
        )

        # inception module 3 part 2
        # 8 * 8 * 2048 -> 8 * 8 * 2048
        # 2048 = 512 + 512 + 512 + 512
        branch1x1_3_2 = self.conv2d_bn(
            inputs=outputs_3_1,
            filters=512,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_2_branch_1x1"
        )

        branch3x3_3_2 = self.conv2d_bn(
            inputs=outputs_3_1,
            filters=512,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_2_branch_3x3"
        )
        branch3x3_3_2_1 = self.conv2d_bn(
            inputs=branch3x3_3_2,
            filters=256,
            kernel_size=(1, 3),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_2_1_branch_3x3"
        )
        branch3x3_3_2_2 = self.conv2d_bn(
            inputs=branch3x3_3_2,
            filters=256,
            kernel_size=(3, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_2_2_branch_3x3"
        )
        branch3x3_3_2 = tf.concat(
            [branch3x3_3_2_1, branch3x3_3_2_2],
            axis=3,
            name='inception_module_3_2_branch_3x3'
        )

        branch3x3dbl_3_2 = self.conv2d_bn(
            inputs=outputs_3_1,
            filters=512,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_2_branch_3x3dbl"
        )
        branch3x3dbl_3_2 = self.conv2d_bn(
            inputs=branch3x3dbl_3_2,
            filters=256,
            kernel_size=(3, 3),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_2_branch_3x3dbl"
        )
        branch3x3dbl_3_2_1 = self.conv2d_bn(
            inputs=branch3x3dbl_3_2,
            filters=256,
            kernel_size=(1, 3),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_2_1_branch_3x3dbl"
        )
        branch3x3dbl_3_2_2 = self.conv2d_bn(
            inputs=branch3x3dbl_3_2,
            filters=256,
            kernel_size=(3, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_2_2_branch_3x3dbl"
        )
        branch3x3dbl_3_2 = tf.concat(
            [branch3x3dbl_3_2_1, branch3x3dbl_3_2_2],
            axis=3,
            name='inception_module_3_2_branch_3x3dbl'
        )

        branch_pool_3_2 = tf.layers.max_pooling2d(
            inputs=outputs_3_1,
            pool_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_2_branch_pool"
        )
        branch_pool_3_2 = self.conv2d_bn(
            inputs=branch_pool_3_2,
            filters=512,
            kernel_size=(1, 1),
            training=self.is_training,
            strides=(1, 1),
            padding="SAME",
            name="inception_module_3_2_branch_pool"
        )

        outputs_3_2 = tf.concat(
            [branch1x1_3_2, branch3x3_3_2, branch3x3dbl_3_2, branch_pool_3_2],
            axis=3,
            name='outputs_3_1'
        )

        outputs_pool = tf.layers.average_pooling2d(
            inputs=outputs_3_2,
            pool_size=(8, 8),
            strides=(1, 1),
            padding="VALID",
            name='outputs_pool'
        )

        self.outputs = tf.layers.dense(
            inputs=outputs_pool,
            units=output_shape,
            activation=tf.nn.softmax,
            name='outputs'
        )

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.y))
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        self.correct_infer = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.outputs, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_infer, tf.float32))

    def train(self, sess, x_train, y_train, x_valid, y_valid, batch_size, epoch):
        for i in range(epoch):
            start_time = time.time()
            for batch_start in range(0, len(x_train), batch_size):
                batch_end = batch_start + batch_size
                batch_x = x_train[batch_start:batch_end]
                batch_y = y_train[batch_start:batch_end]

                sess.run(self.train_op, feed_dict={self.x: batch_x, self.y: batch_y, self.is_training: True})

            loss_value = sess.run(
                self.loss,
                feed_dict={
                    self.x: batch_x,
                    self.y: batch_y,
                    self.is_training: False
                })
            accuracy_value = sess.run(
                self.accuracy,
                feed_dict={
                    self.x: x_valid,
                    self.y: y_valid,
                    self.is_training: False
                })
            print('-------------%d次迭代---------------' % i)
            print('总训练时间：%.2f' % (time.time() - start_time))
            print('当前损失值：%.2f' % loss_value)
            print('当前准确率：%.2f' % accuracy_value)
        print('-------------训练结束---------------')

    def inference(self, sess, inputs):
        return sess.run(tf.argmax(self.outputs, axis=1), feed_dict={self.x: inputs})

    @staticmethod
    def conv2d_bn(inputs, filters, kernel_size, training, strides=(1, 1), padding='SAME', name=None):
        outputs = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None
        )
        outputs = tf.layers.batch_normalization(outputs, training=training)
        outputs = tf.nn.relu(outputs, name=name)

        return outputs
