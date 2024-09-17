from keras import  backend as K

class Config:

    def __init__(self):

        self.anchorBoxScales = [128, 256, 512]
        self.anchorBoxRatios = [[1, 1], [1, 2], [2, 1]]
        self.rpnStride = 16
        self.numRois = 32
        self.verbose = True
        self.modelPath = "logs/model.h5"
        self.rpnMinOverlap = 0.3
        self.rpnMaxOverlap = 0.7
        self.classifierMinOverlap = 0.1
        self.classifierMaxOverlap = 0.5
        self.classifierRegrStd = [8.0, 8.0, 4.0, 4.0]