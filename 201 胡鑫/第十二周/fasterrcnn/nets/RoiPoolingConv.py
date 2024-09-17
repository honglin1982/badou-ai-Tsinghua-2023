from keras.engine.topology import Layer
import keras.backend as k

if k.backend() == 'tensorflow':
    import tensorflow as tf


# noinspection PyCompatibility
class RoiPoolingConv(Layer):

    def __init__(self, pool_size, num_rois, **kwargs):
        super(RoiPoolingConv, self).__init__(**kwargs)
        self.nb_channels = None
        self.dim_ordering = k.set_image_data_format("channels_last")

        self.pool_size = pool_size
        self.num_rois = num_rois

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert (len(x) == 2)
        # roi_input:[None,None,4],第一个None表示batch_size, 第二个None表示
        # 一张图片有多少个建议框,4表示这些建议框的的坐标
        img = x[0]
        rois = x[1]

        outputs = []

        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = k.cast(x, 'int32')
            y = k.cast(y, 'int32')
            w = k.cast(w, 'int32')
            h = k.cast(h, 'int32')
            # 只将框框住的部分roipool
            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = k.concatenate(outputs, axis=0)
        final_output = k.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        final_output = k.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
