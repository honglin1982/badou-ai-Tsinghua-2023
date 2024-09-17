from keras.engine.topology import  Layer
import keras.backend as K

if K.backend() == 'tensorflow':
    import tensorflow as tf

class RoiPoolingConv(Layer):

    def __init__(self, poolSize, numRois, **kwargs):

        self.dimOrdering = K.image_data_format()
        assert self.dimOrdering in {'channels_first', 'channels_last'}, 'dimOrdering must be in {channelsFirst, channesLast}'
        self.poolSize = poolSize
        self.numRois = numRois
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, inputShape):
        self.nbChannels = inputShape[0][3]

    def compute_output_shape(self, inputShape):
        return None, self.numRois, self.poolSize, self.poolSize, self.nbChannels

    def call(self, x, mask=None):

        assert(len(x) == 2)
        img, rois = x[0], x[1]
        outputs = []
        for roiIdx in range(self.numRois):

            x = rois[0, roiIdx, 0]
            y = rois[0, roiIdx, 1]
            w = rois[0, roiIdx, 2]
            h = rois[0, roiIdx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            rs = tf.image.resize_images(img[:, y : y + h, x : x + w, :],
                                        (self.poolSize, self.poolSize))
            outputs.append(rs)
        finalOutput = K.concatenate(outputs, axis=0)
        finalOutput = K.reshape(finalOutput, (1, self.numRois, self.poolSize,
                                              self.poolSize, self.nbChannels))
        finalOutput = K.permute_dimensions(finalOutput, (0, 1, 2, 3, 4))
        return finalOutput
