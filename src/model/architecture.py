import tensorflow as tf
from tensorflow.keras import models, layers, regularizers, Input, Model


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

    # todo


class ReflectionPadding3D(layers.Layer):
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1],
                input_shape[3] + 2 * self.padding[2],
                input_shape[4])

    def call(self, input_tensor):
        padding_depth, padding_height, padding_width = self.padding
        return tf.pad(input_tensor,
                      tf.constant([[0, 0],
                                   [padding_depth, padding_depth],
                                   [padding_height, padding_height],
                                   [padding_width, padding_width],
                                   [0, 0]]),
                      'SYMMETRIC')


class SFTLayer(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), **kwargs):
        super(SFTLayer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel = kernel_size
        # todo: are those weights trainable? https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
        self.gamma_conv = layers.Conv2D(self.filters, self.kernel, activation=layers.LeakyReLU(), padding='same')
        self.beta_conv = layers.Conv2D(self.filters, self.kernel, activation=layers.LeakyReLU(), padding='same')
        self.x_shape = None

    def build(self, input_shape):
        self.x_shape = input_shape[0]

    def call(self, inputs):
        x, psi = inputs  # psi is input from detail branch; x is input approx branch
        gamma = self.gamma_conv(psi)
        beta = self.beta_conv(psi)
        merged = None

        for band_no in range(self.x_shape[-2]):
            x_band = x[:, :, :, band_no, :]
            # todo: check if this works as intended
            # todo: restart from here:
            # build a branch that extracts edged (maybe from sentinel input only) and that injects them into the other branch
            # test this and monitor the output with a very small training sample to see if it actually works
            # also rethink the dimensions of each "cube" in a layer. the paper says k*s*s*c where c is the number of bands
            # also print model graph
            # and take a look at test_model.py
            x_band = gamma * x_band + beta
            x_band = tf.expand_dims(x_band, axis=-2)
            if merged is None:
                merged = x_band
            else:
                merged = tf.concat([merged, x_band], axis=-2)

        return merged


class SaPnn:
    def __init__(self, tile_size, no_input_bands, no_output_bands):
        self.tile_size = tile_size
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.kernel2d = (7, 7)
        self.padding2d = (self.kernel2d[0] // 2, self.kernel2d[1] // 2)
        self.kernel3d = (7, 7, 3)
        self.padding3d = (self.kernel3d[0] // 2, self.kernel3d[1] // 2, self.kernel3d[2] // 2)
        self.feature_maps = 32
        self.model = None
        self.create_layers()

    def create_layers(self):
        # first layer
        input_detail = Input(shape=(self.tile_size, self.tile_size, self.no_input_bands), name='x')
        detail = ReflectionPadding2D(padding=self.padding2d)(input_detail)
        detail = layers.Conv2D(self.feature_maps, self.kernel2d, padding='valid', activation='relu')(detail)
        input_approx = Input(shape=(self.tile_size, self.tile_size, self.no_output_bands), name='x1')
        approx = tf.expand_dims(input_approx, axis=-1)
        approx = ReflectionPadding3D(padding=self.padding3d)(approx)
        approx = layers.Conv3D(self.feature_maps, self.kernel3d, padding='valid', activation='relu')(approx)
        merged_branches = SFTLayer(filters=self.feature_maps)([approx, detail])

        # second layer
        detail = ReflectionPadding2D(padding=self.padding2d)(detail)
        detail = layers.Conv2D(self.feature_maps, self.kernel2d, padding='valid', activation='relu')(detail)
        approx = ReflectionPadding3D(padding=self.padding3d)(merged_branches)
        approx = layers.Conv3D(self.feature_maps, self.kernel3d, padding='valid', activation='relu')(approx)
        sft_layer = SFTLayer(filters=self.feature_maps)
        merged_branches = sft_layer([approx, detail])

        # third layer
        detail = ReflectionPadding2D(padding=self.padding2d)(detail)
        detail = layers.Conv2D(self.feature_maps, self.kernel2d, padding='valid', activation='relu')(detail)
        approx = ReflectionPadding3D(padding=self.padding3d)(merged_branches)
        approx = layers.Conv3D(self.feature_maps, self.kernel3d, padding='valid', activation='relu')(approx)
        sft_layer = SFTLayer(filters=self.feature_maps)
        merged_branches = sft_layer([approx, detail])

        y = layers.Conv3D(1, (1, 1, 1), padding='valid', activation='linear')(merged_branches)
        y = tf.squeeze(y, axis=-1)

        self.model = Model(inputs=[input_detail, input_approx], outputs=y)


class TestSaPnn:
    def __init__(self, tile_size, no_input_bands, no_output_bands):
        self.tile_size = tile_size
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.kernel2d = (3, 3)
        self.padding2d = (self.kernel2d[0] // 2, self.kernel2d[1] // 2)
        self.kernel3d = (7, 7, 3)
        self.padding3d = (self.kernel3d[0] // 2, self.kernel3d[1] // 2, self.kernel3d[2] // 2)
        self.feature_maps = 16
        self.model = None
        self.create_layers()

    def create_layers(self):
        # first layer
        input_detail = Input(shape=(self.tile_size, self.tile_size, self.no_input_bands), name='x')
        detail = ReflectionPadding2D(padding=self.padding2d)(input_detail)
        detail = layers.Conv2D(self.feature_maps, self.kernel2d, padding='valid', activation='relu')(detail)
        input_approx = Input(shape=(self.tile_size, self.tile_size, self.no_output_bands), name='x1')
        approx = tf.expand_dims(input_approx, axis=-1)
        approx = ReflectionPadding3D(padding=self.padding3d)(approx)
        approx = layers.Conv3D(self.feature_maps, self.kernel3d, padding='valid', activation='relu')(approx)
        merged_branches = SFTLayer(filters=self.feature_maps)([approx, detail])

        y = layers.Conv3D(1, (1, 1, 1), padding='valid', activation='linear')(merged_branches)
        y = tf.squeeze(y, axis=-1)

        self.model = Model(inputs=[input_detail, input_approx], outputs=y)


class FCNN:
    """
    https://www.mdpi.com/2072-4292/9/11/1139#
    Repo: https://github.com/MeiShaohui/Hyperspectral-Image-Spatial-Super-Resolution-via-3D-Full-Convolutional-Neural-Network/blob/master/network3d.py
    """

    def __init__(self, tile_size, no_input_bands, no_output_bands):
        self.tile_size = tile_size
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.model = None
        self.create_layers()

    def create_layers(self):
        seed_gen = tf.keras.utils.set_random_seed(42)
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=seed_gen)

        # first layer
        input3d = Input(shape=(self.tile_size, self.tile_size, self.no_input_bands, 1), name='x1')
        conv1 = layers.Conv3D(64, (9, 9, 7), padding='same',
                              activation='relu',
                              kernel_initializer=initializer)(input3d)
        conv2 = layers.Conv3D(32, (1, 1, 1), padding='same',
                              activation='relu',
                              kernel_initializer=initializer)(conv1)
        conv3 = layers.Conv3D(9, (1, 1, 1), padding='same',
                              activation='relu',
                              kernel_initializer=initializer)(conv2)
        y = layers.Conv3D(1, (5, 5, 3), padding='same',
                          activation='linear',
                          kernel_initializer=initializer)(conv3)

        self.model = Model(inputs=input3d, outputs=y)


class TestFCNN:
    def __init__(self, tile_size, no_input_bands, no_output_bands):
        self.tile_size = tile_size
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.model = None
        self.create_layers()

    def create_layers(self):
        seed_gen = tf.keras.utils.set_random_seed(42)
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=seed_gen)

        # first layer
        # input3d = Input(shape=(self.tile_size, self.tile_size, self.no_input_bands, 1), name='x1')
        input2d = Input(shape=(self.tile_size, self.tile_size, self.no_input_bands), name='x1')
        # input2d = tf.squeeze(input3d, axis=-1)
        conv1 = layers.Conv2D(64, (9, 9), padding='same',
                              activation='relu',
                              kernel_initializer=initializer)(input2d)
        conv2 = layers.Conv2D(32, (1, 1), padding='same',
                              activation='relu',
                              kernel_initializer=initializer)(conv1)
        y = layers.Conv2D(self.no_output_bands, (3, 3), padding='same',
                              activation='relu',
                              kernel_initializer=initializer)(conv2)
        # y = tf.expand_dims(conv3, axis=-1)

        # todo: vgl mit Masi mit 20 Bändern --> dimension expand könnte Problem sein

        self.model = Model(inputs=input2d, outputs=y)
