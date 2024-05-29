import tensorflow as tf
from tensorflow.keras import models, layers, regularizers, Input


class ReflectionPadding2D(layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1],
                input_shape[3])

    def call(self, input_tensor):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor,
                      tf.constant([[0, 0],
                                   [padding_height, padding_height],
                                   [padding_width, padding_width],
                                   [0, 0]]),
                      'SYMMETRIC')

class Masi:
    def __init__(self, tile_size, no_input_bands, no_output_bands):
        self.tile_size = tile_size
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.model = models.Sequential()
        self.create_layers()
    def create_layers(self):
        input_shape = (self.tile_size, self.tile_size, self.no_input_bands)
        self.model.add(Input(shape=input_shape))

        kernel = (9, 9)
        padding = (lambda x: (x[0] // 2, x[1] // 2))
        self.model.add(ReflectionPadding2D(padding=padding(kernel)))
        self.model.add(layers.Conv2D(64,
                                kernel,
                                # activation=tf.keras.layers.LeakyReLU(),
                                activation='relu',
                                kernel_regularizer=regularizers.l1(0.015),
                                padding='valid'))

        kernel = (3, 3)
        self.model.add(ReflectionPadding2D(padding=padding(kernel)))
        self.model.add(layers.Conv2D(32,
                                kernel,
                                # activation=tf.keras.layers.LeakyReLU(),
                                activation='relu',
                                kernel_regularizer=regularizers.l1(0.03),
                                padding='valid'))

        kernel = (3, 3)
        self.model.add(ReflectionPadding2D(padding=padding(kernel)))
        self.model.add(layers.Conv2D(self.no_output_bands,
                                kernel,
                                activation='linear',
                                kernel_regularizer=regularizers.l1(0.015),
                                padding='valid'))

class Brook:
    def __init__(self, tile_size, no_input_bands, no_output_bands):
        self.tile_size = tile_size
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.model = models.Sequential()
        self.create_layers()

    # todo
    def create_layers(self):
        input_shape = (self.tile_size, self.tile_size, self.no_input_bands)
        self.model.add(Input(shape=input_shape))

        kernel = (9, 9)
        padding = (lambda x: (x[0] // 2, x[1] // 2))
        self.model.add(ReflectionPadding2D(padding=padding(kernel)))
        self.model.add(layers.Conv2D(64,
                                     kernel,
                                     # activation=tf.keras.layers.LeakyReLU(),
                                     activation='relu',
                                     kernel_regularizer=regularizers.l1(0.015),
                                     padding='valid'))