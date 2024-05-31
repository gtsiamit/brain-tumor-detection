from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, concatenate, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2

# Model parameters
INIT_FEATURES = 16


def build_unet_model(input_shape: tuple):
    """Builds a UNET segmentation model

    Args:
        input_shape (tuple): Model input shape

    Returns:
        model (tf.keras.Model): UNET model
    """

	# Input Layer
    input_layer = Input(shape=input_shape)

    # 1st contracting block
    down_conv1 = Conv2D(filters=INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(input_layer)
    down_conv1 = BatchNormalization()(down_conv1)
    down_conv1 = Activation('relu')(down_conv1)
    down_conv1 = Conv2D(filters=INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(down_conv1)
    down_conv1 = BatchNormalization()(down_conv1)
    down_conv1 = Activation('relu')(down_conv1)
    down_pool1 = MaxPooling2D(pool_size=(2,2), strides=None, padding='valid')(down_conv1)
    down_pool1 = Dropout(rate = 0.1)(down_pool1)

    # 2nd contracting block
    down_conv2 = Conv2D(filters=2*INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(down_pool1)
    down_conv2 = BatchNormalization()(down_conv2)
    down_conv2 = Activation('relu')(down_conv2)
    down_conv2 = Conv2D(filters=2*INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(down_conv2)
    down_conv2 = BatchNormalization()(down_conv2)
    down_conv2 = Activation('relu')(down_conv2)
    down_pool2 = MaxPooling2D(pool_size=(2,2), strides=None, padding='valid')(down_conv2)
    down_pool2 = Dropout(rate = 0.1)(down_pool2)

    # 3rd contracting block
    down_conv3 = Conv2D(filters=4*INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(down_pool2)
    down_conv3 = BatchNormalization()(down_conv3)
    down_conv3 = Activation('relu')(down_conv3)
    down_conv3 = Conv2D(filters=4*INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(down_conv3)
    down_conv3 = BatchNormalization()(down_conv3)
    down_conv3 = Activation('relu')(down_conv3)
    down_pool3 = MaxPooling2D(pool_size=(2,2), strides=None, padding='valid')(down_conv3)
    down_pool3 = Dropout(rate = 0.1)(down_pool3)

    # 4th contracting block
    down_conv4 = Conv2D(filters=8*INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(down_pool3)
    down_conv4 = BatchNormalization()(down_conv4)
    down_conv4 = Activation('relu')(down_conv4)
    down_conv4 = Conv2D(filters=8*INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(down_conv4)
    down_conv4 = BatchNormalization()(down_conv4)
    down_conv4 = Activation('relu')(down_conv4)
    down_pool4 = MaxPooling2D(pool_size=(2,2), strides=None, padding='valid')(down_conv4)
    down_pool4 = Dropout(rate = 0.1)(down_pool4)

    # Center block
    center_conv1 = Conv2D(filters=16*INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(down_pool4)
    center_conv1 = BatchNormalization()(center_conv1)
    center_conv1 = Activation('relu')(center_conv1)
    center_conv2 = Conv2D(filters=16*INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(center_conv1)
    center_conv2 = BatchNormalization()(center_conv2)
    center_conv2 = Activation('relu')(center_conv2)

    # 4th expansive block
    up_conv4 = Conv2D(filters=8*INIT_FEATURES, kernel_size=(2,2), strides=(1,1), padding='same')(UpSampling2D(size=(2,2))(center_conv2))
    up_conv4 = concatenate([up_conv4, down_conv4], axis=-1)
    up_conv4 = Dropout(rate = 0.1)(up_conv4)
    up_conv4 = Conv2D(filters=8*INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(up_conv4)
    up_conv4 = BatchNormalization()(up_conv4)
    up_conv4 = Activation('relu')(up_conv4)
    up_conv4 = Conv2D(filters=8*INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(up_conv4)
    up_conv4 = BatchNormalization()(up_conv4)
    up_conv4 = Activation('relu')(up_conv4)

    # 3rd expansive block
    up_conv3 = Conv2D(filters=4*INIT_FEATURES, kernel_size=(2,2), strides=(1,1), padding='same')(UpSampling2D(size=(2,2))(up_conv4))
    up_conv3 = concatenate([up_conv3, down_conv3], axis=-1)
    up_conv3 = Dropout(rate = 0.1)(up_conv3)
    up_conv3 = Conv2D(filters=4*INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(up_conv3)
    up_conv3 = BatchNormalization()(up_conv3)
    up_conv3 = Activation('relu')(up_conv3)
    up_conv3 = Conv2D(filters=4*INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(up_conv3)
    up_conv3 = BatchNormalization()(up_conv3)
    up_conv3 = Activation('relu')(up_conv3)

    # 2nd expansive block
    up_conv2 = Conv2D(filters=2*INIT_FEATURES, kernel_size=(2,2), strides=(1,1), padding='same')(UpSampling2D(size=(2,2))(up_conv3))
    up_conv2 = concatenate([up_conv2, down_conv2], axis=-1)
    up_conv2 = Dropout(rate = 0.1)(up_conv2)
    up_conv2 = Conv2D(filters=2*INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(up_conv2)
    up_conv2 = BatchNormalization()(up_conv2)
    up_conv2 = Activation('relu')(up_conv2)
    up_conv2 = Conv2D(filters=2*INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(up_conv2)
    up_conv2 = BatchNormalization()(up_conv2)
    up_conv2 = Activation('relu')(up_conv2)

    # 1st expansive block
    up_conv1 = Conv2D(filters=INIT_FEATURES, kernel_size=(2,2), strides=(1,1), padding='same')(UpSampling2D(size=(2,2))(up_conv2))
    up_conv1 = concatenate([up_conv1, down_conv1], axis=-1)
    up_conv1 = Dropout(rate = 0.1)(up_conv1)
    up_conv1 = Conv2D(filters=INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(up_conv1)
    up_conv1 = BatchNormalization()(up_conv1)
    up_conv1 = Activation('relu')(up_conv1)
    up_conv1 = Conv2D(filters=INIT_FEATURES, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(up_conv1)
    up_conv1 = BatchNormalization()(up_conv1)
    up_conv1 = Activation('relu')(up_conv1)

    output_layer = Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding='valid', activation="sigmoid")(up_conv1)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    optimizer = RMSprop(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics = ['accuracy'])

    return model

