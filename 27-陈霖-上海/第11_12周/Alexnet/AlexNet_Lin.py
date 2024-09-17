from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization

def AlexNet(input_shape=(224,224,3),output_shape=2):
    model = Sequential()
    # First layer: conv=[strides = 4,size=11],output=96,output_shape=(55,55,96)
    model.add(
        Conv2D(
            filters = 48,
            kernel_size=(11,11),
            strides=(4,4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )
    # Second layer: conv=[strides = 1,size=5],output=256,output_shape=(27,27,256)
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    model.add(BatchNormalization())
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    # Third layer: conv=[strides = 1,size=3],output=384,output_shape=(13,13,384)
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # Fourth layer: conv=[strides = 1,size=3],output=256,output_shape=(13,13,256)
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    # Fifth layer: Full connect
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_shape, activation='softmax'))

    return model
