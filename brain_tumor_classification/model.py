from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    InputLayer,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop


def build_model(input_shape: tuple):
    """CNN classification model

    Args:
            input_shape (tuple): Input shape of model

    Returns:
            model (tensorflow.keras.Model): CNN model
    """

    # Create model
    model = Sequential()

    # Input Layer
    model.add(InputLayer(input_shape=input_shape))

    # 1st Convolution Layer
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(2, 2),
            strides=(1, 1),
            padding="valid",
            activation="relu",
            kernel_regularizer=l2(0.001),
            bias_regularizer=l2(0.001),
        )
    )
    # 1st Batch normalization Layer
    model.add(BatchNormalization())
    # 1st Max pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"))

    # 2nd Convolution Layer
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(2, 2),
            strides=(1, 1),
            padding="valid",
            activation="relu",
            kernel_regularizer=l2(0.001),
            bias_regularizer=l2(0.001),
        )
    )
    # 2nd Batch normalization Layer
    model.add(BatchNormalization())
    # 2nd Max pooling
    model.add(
        MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")
    )  # with strides=None, strides defaults to pool size
    model.add(Dropout(rate=0.2))

    # Flatten Layer
    model.add(Flatten())

    # Dense Layers
    model.add(
        Dense(
            units=512,
            activation="relu",
            kernel_regularizer=l2(0.001),
            bias_regularizer=l2(0.001),
        )
    )

    # Output Dense Layer
    model.add(Dense(1, activation="sigmoid"))

    optimizer = RMSprop(learning_rate=0.0001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model
