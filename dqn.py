import tensorflow as tf
from tensorflow.keras import layers


def make_dqn(actions, input_dimensions):
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(32, (8, 8), strides=(2, 2), padding='same', input_shape=input_dimensions))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2D(64, (4, 4), strides=(1, 1), padding='same'))
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(actions, activation='linear'))
    
    return model
