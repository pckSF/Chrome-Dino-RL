import tensorflow as tf
from tensorflow.keras import layers


def make_dqn(actions, input_dimensions):
    model = tf.keras.Sequential()

    model.add(
        layers.Conv2D(
            16, (8, 8), strides=(4, 4), padding="same", input_shape=input_dimensions
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2D(32, (4, 4), strides=(2, 2), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Flatten())

    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(actions, activation="linear"))

    return model
