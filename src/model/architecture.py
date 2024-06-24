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
        self.name = 'Masi'
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
        self.gamma_conv_2 = layers.Conv2D(self.filters, self.kernel, activation=layers.LeakyReLU(), padding='same')
        self.beta_conv = layers.Conv2D(self.filters, self.kernel, activation=layers.LeakyReLU(), padding='same')
        self.beta_conv_2 = layers.Conv2D(self.filters, self.kernel, activation=layers.LeakyReLU(), padding='same')
        self.x_shape = None

    def build(self, input_shape):
        self.x_shape = input_shape[0]

    def call(self, inputs):
        x, psi = inputs  # psi is input from detail branch; x is input approx branch
        gamma_conv_1 = self.gamma_conv(psi)
        gamma = self.gamma_conv_2(gamma_conv_1)
        beta_conv_1 = self.beta_conv(psi)
        beta = self.beta_conv_2(beta_conv_1)
        merged = None

        for band_no in range(self.x_shape[-2]):
            x_band = x[:, :, :, band_no, :]
            # todo: check if this works as intended
            # build a branch that extracts edged (maybe from sentinel input only) and that injects them into the other branch
            # test this and monitor the output with a very small training sample to see if it actually works
            # also rethink the dimensions of each "cube" in a layer. the paper says k*s*s*c where c is the number of bands
            # also print model graph
            # and take a look at test_model.py
            x_band = layers.Multiply()([x_band, gamma])
            x_band = layers.Add()([x_band, beta])
            # x_band = gamma * x_band + beta
            x_band = tf.expand_dims(x_band, axis=-2)
            if merged is None:
                merged = x_band
            else:
                merged = tf.concat([merged, x_band], axis=-2)

        return merged


class SaPNN:
    def __init__(self, tile_size, no_input_bands, no_output_bands):
        self.name = 'SaPNN'
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


class TestSaPNN:
    def __init__(self, tile_size, no_input_bands, no_output_bands):
        self.name = 'TestSaPNN'
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
        padding2d = (lambda x: (x[0] // 2, x[1] // 2))
        kernel = (7, 7)
        # detail_1_pad = ReflectionPadding2D(padding=padding2d(kernel))(input_detail)
        # detail_1 = layers.Conv2D(64, kernel, padding='same', activation='relu')(input_detail)

        input_approx = Input(shape=(self.tile_size, self.tile_size, self.no_output_bands, 1), name='x1')
        padding3d = (lambda x: (x[0] // 2, x[1] // 2, x[2] // 2))
        kernel = (9, 9, 7)
        # approx_1_pad = ReflectionPadding3D(padding=padding3d(kernel))(input_approx)
        approx_1 = layers.Conv3D(64, kernel, padding='same', activation='relu')(input_approx)
        # approx_skipped = layers.Add()([input_approx, approx_1])
        merged_branches = SFTLayer(filters=64)([approx_1, input_detail])

        # # second layer
        # kernel = (1, 1, 1)
        # approx_2_pad = ReflectionPadding3D(padding=padding3d(kernel))(merged_branches)
        # approx_2 = layers.Conv3D(32, kernel, padding='valid', activation='relu')(approx_2_pad)
        #
        # kernel = (3, 3)
        # detail_2_pad = ReflectionPadding2D(padding=padding2d(kernel))(detail_1)
        # detail_2 = layers.Conv2D(32, kernel, padding='valid', activation='relu')(detail_2_pad)
        # merged_branches = SFTLayer(filters=32)([approx_2, detail_2])

        # # third layer
        # kernel = (1, 1, 1)
        # approx_3_pad = ReflectionPadding3D(padding=padding(kernel))(approx_2)
        # approx_3 = layers.Conv3D(9, kernel, padding='valid', activation='relu')(approx_3_pad)

        # convOutput = layers.Conv3D(1, (5, 5, 3), padding='same', activation='linear')(approx_3)
        convOutput = layers.Conv3D(1, (5, 5, 3), padding='same', activation='linear')(merged_branches)
        y = tf.squeeze(convOutput, axis=-1)

        self.model = Model(inputs=[input_detail, input_approx], outputs=y)
        # todo: starts to fit (a little) with modified sft layer (double conv2d) and 6 input + 3 output bands
        # todo: test the trained model and add more layers, test this with Pavia dataset (only matlab file!)
        # atm with a second layer it doesnt fit anymore
        # todo: kernel size very important! fcnn doesnt fit with all kernels = (7,7,3)


class FCNN:
    """
    https://www.mdpi.com/2072-4292/9/11/1139#
    Repo: https://github.com/MeiShaohui/Hyperspectral-Image-Spatial-Super-Resolution-via-3D-Full-Convolutional-Neural-Network/blob/master/network3d.py
    """

    def __init__(self, tile_size, no_input_bands, no_output_bands):
        self.name = 'FCNN'
        self.tile_size = tile_size
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.model = None
        self.create_layers()

    def create_layers(self):
        # seed_gen = tf.keras.utils.set_random_seed(42)
        # initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=seed_gen)

        # first layer
        input3d = Input(shape=(self.tile_size, self.tile_size, self.no_output_bands, 1), name='x1')
        conv1 = layers.Conv3D(64, (9, 9, 7), padding='same',
                              activation='relu')(input3d)
        conv2 = layers.Conv3D(32, (1, 1, 1), padding='same',
                              activation='relu')(conv1)
        conv3 = layers.Conv3D(9, (1, 1, 1), padding='same',
                              activation='relu')(conv2)
        convOut = layers.Conv3D(1, (5, 5, 3), padding='same',
                                activation='linear')(conv3)
        y = tf.squeeze(convOut, axis=-1)

        self.model = Model(inputs=input3d, outputs=y)

        # todo: restart from here
        # fcnn with 4d input and 3d output works for 3 input and ouput bands
        # todo: train with more bands (and more training samples)
        # atm fits with 40 bands but not with 224 --> continue at TestSaPNN
        # todo: Verschiebungen beim Resamplen fixen!


class TestFCNN:
    def __init__(self, tile_size, no_input_bands, no_output_bands):
        self.name = 'TestFCNN'
        self.tile_size = tile_size
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.model = None
        self.create_layers()

    def create_layers(self):
        seed_gen = tf.keras.utils.set_random_seed(42)
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=seed_gen)

        # first layer
        input3d = Input(shape=(self.tile_size, self.tile_size, self.no_input_bands, 1), name='x')
        # input2d = Input(shape=(self.tile_size, self.tile_size, self.no_input_bands), name='x')
        input2d = tf.squeeze(input3d, axis=-1)
        kernel = (9, 9)
        padding = (lambda x: (x[0] // 2, x[1] // 2))
        reflect_pad_1 = ReflectionPadding2D(padding=padding(kernel))(input2d)
        conv1 = layers.Conv2D(64, kernel, padding='valid',
                              activation='relu',
                              kernel_regularizer=regularizers.l1(0.015))(reflect_pad_1)
        kernel = (3, 3)
        reflect_pad_2 = ReflectionPadding2D(padding=padding(kernel))(conv1)
        conv2 = layers.Conv2D(32, kernel, padding='valid',
                              activation='relu',
                              kernel_regularizer=regularizers.l1(0.03))(reflect_pad_2)
        kernel = (3, 3)
        reflect_pad_2 = ReflectionPadding2D(padding=padding(kernel))(conv2)
        y = layers.Conv2D(self.no_output_bands, kernel, padding='valid',
                          activation='linear',
                          kernel_regularizer=regularizers.l1(0.015))(reflect_pad_2)

        self.model = Model(inputs=input3d, outputs=y)
