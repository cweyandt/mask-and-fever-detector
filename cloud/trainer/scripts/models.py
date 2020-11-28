from tensorflow.keras.layers import (
    Input,
    Conv2D,
    AveragePooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# create a model
def create_mask_detector_mobilenet(input_shape):
    # use mobilenetv2 as a feature extractor
    input_layer = Input(input_shape)  # (X,Y,channel)
    mobilenetv2 = MobileNetV2(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False,
        input_tensor=input_layer,
    )

    # freeze the weights in mobilenetv2
    for layer in mobilenetv2.layers:
        layer.trainable = False

    # let's turn it into a classifier
    X = mobilenetv2.output
    #X = AveragePooling2D(pool_size=(7, 7))(X) for 224,224
    X = AveragePooling2D(pool_size=(3, 3))(X) #for 96,96
    X = Flatten()(X)
    X = Dense(512, activation="relu")(X)
    X = Dropout(0.4)(X)
    X = Dense(256, activation="relu")(X)
    X = Dense(2, activation="softmax")(X)

    return Model(inputs=input_layer, outputs=X)
