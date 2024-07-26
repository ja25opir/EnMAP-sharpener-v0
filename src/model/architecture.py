import tensorflow as tf
from tensorflow.keras import models, layers, regularizers, Input, Model


@tf.keras.utils.register_keras_serializable()
def ms_ssim_l1_loss(y_true, y_pred):
    # alpha * (1 - MS_SSIM) + (1 - alpha) * L1_loss
    # paper: https://arxiv.org/pdf/1511.08861
    max_raster_value = 10000
    alpha = 0.84

    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred)) # == mean absolute error
    ms_ssim_loss = (1 - tf.image.ssim(y_true, y_pred, max_raster_value))

    loss = (alpha * ms_ssim_loss + (1 - alpha) * l1_loss)

    return tf.reduce_mean(loss)

    # l1 only
    # return tf.reduce_mean(tf.abs(y_true - y_pred))


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
                      'REFLECT')


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
                      'REFLECT')


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
            # todo: check if this works as intended (operation in every band, but also every feature map?)
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
        detail = layers.Conv2D(self.feature_maps, self.kernel2d, padding='same', activation='relu')(input_detail)
        input_approx = Input(shape=(self.tile_size, self.tile_size, self.no_output_bands), name='x1')
        approx = tf.expand_dims(input_approx, axis=-1)
        approx = layers.Conv3D(self.feature_maps, self.kernel3d, padding='same', activation='relu')(approx)
        merged_branches = SFTLayer(filters=self.feature_maps)([approx, detail])

        # second layer
        detail = layers.Conv2D(self.feature_maps, self.kernel2d, padding='same', activation='relu')(detail)
        approx = layers.Conv3D(self.feature_maps, self.kernel3d, padding='same', activation='relu')(merged_branches)
        sft_layer = SFTLayer(filters=self.feature_maps)
        merged_branches = sft_layer([approx, detail])

        # third layer
        detail = layers.Conv2D(self.feature_maps, self.kernel2d, padding='same', activation='relu')(detail)
        approx = layers.Conv3D(self.feature_maps, self.kernel3d, padding='same', activation='relu')(merged_branches)
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
        detail_1 = layers.Conv2D(64, kernel, padding='same', activation='relu')(input_detail)

        input_approx = Input(shape=(self.tile_size, self.tile_size, self.no_output_bands, 1), name='x1')
        padding3d = (lambda x: (x[0] // 2, x[1] // 2, x[2] // 2))
        kernel = (9, 9, 7)
        # approx_1_pad = ReflectionPadding3D(padding=padding3d(kernel))(input_approx)
        approx_1 = layers.Conv3D(64, kernel, padding='same', activation='relu')(input_approx)
        # approx_skipped = layers.Add()([input_approx, approx_1])
        merged_branches = SFTLayer(filters=64)([approx_1, detail_1])

        # second layer
        kernel = (1, 1, 1)
        # approx_2_pad = ReflectionPadding3D(padding=padding3d(kernel))(merged_branches)
        approx_2 = layers.Conv3D(32, kernel, padding='same', activation='relu')(merged_branches)

        kernel = (3, 3)
        # detail_2_pad = ReflectionPadding2D(padding=padding2d(kernel))(detail_1)
        detail_2 = layers.Conv2D(32, kernel, padding='same', activation='relu')(detail_1)
        merged_branches = SFTLayer(filters=32)([approx_2, detail_2])

        # # third layer
        kernel = (1, 1, 1)
        # approx_3_pad = ReflectionPadding3D(padding=padding(kernel))(approx_2)
        approx_3 = layers.Conv3D(9, kernel, padding='same', activation='relu')(merged_branches)

        kernel = (5, 5)
        detail_3 = layers.Conv2D(9, kernel, padding='same', activation='relu')(detail_2)
        merged_branches = SFTLayer(filters=9)([approx_3, detail_3])

        # convOutput = layers.Conv3D(1, (5, 5, 3), padding='same', activation='linear')(approx_3)
        convOutput = layers.Conv3D(1, (5, 5, 3), padding='same', activation='linear')(merged_branches)
        y = tf.squeeze(convOutput, axis=-1)

        self.model = Model(inputs=[input_detail, input_approx], outputs=y)
        # todo: starts to fit (a little) with ONE layer and modified sft layer (double conv2d) and 6 input + 3 output bands
        # todo: more layers --> no fit
        # todo: 224 bands with SaPNN --> memory exceeds
        # todo: train with a single scene without any clouds (Pavia is also only one scene)
        # todo: kernel size very important! fcnn doesnt fit with all kernels = (7,7,3)


class DILayer(layers.Layer):
    """detail injection layer"""

    def __init__(self, kernel_size=(3, 3), **kwargs):
        super(DILayer, self).__init__(**kwargs)
        self.kernel = kernel_size
        self.x_shape = None

    def build(self, input_shape):
        self.x_shape = input_shape[0]

    def call(self, inputs):
        x, edges = inputs
        merged = None

        """Add() edges to each input feature map"""
        # for band_no in range(self.x_shape[-2]):
        #     x_band = x[:, :, :, band_no, :]
        #     for feature_map in range(x_band.shape[-1]):
        #         # x_band = layers.Multiply()([x_band, edges])
        #         x_band = layers.Add()([x_band, edges])
        #     x_band = tf.expand_dims(x_band, axis=-2) # todo: check this
        #     if merged is None:
        #         merged = x_band
        #     else:
        #         merged = tf.concat([merged, x_band], axis=-2)

        """stack edges feature map(s) to input feature maps"""
        for band_no in range(self.x_shape[-2]):
            x_band = x[:, :, :, band_no, :]
            x_stacked = tf.concat([x_band, edges], axis=-1)
            x_stacked = tf.expand_dims(x_stacked, axis=-2)
            if merged is None:
                merged = x_stacked
            else:
                merged = tf.concat([merged, x_stacked], axis=-2)

        return merged


class MMSRes:
    """full custom network"""

    def __init__(self, tile_size, no_input_bands, no_output_bands):
        self.name = 'MMSRes'
        self.tile_size = tile_size
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.padding2d = (lambda x: (x[0] // 2, x[1] // 2))
        self.model = None
        self.create_layers()

    def create_layers(self):
        # todo: restart from here
        # todo: batch normalization as described in https://www.mdpi.com/2076-3417/11/1/288
        #  --> this does not help in main branch (but looks reasonable in 2d branch)
        # todo: test SaPNN (or others) with current data
        # todo: (0) implement evaluation method besides plotting
        # todo: (0.1) prevent negative prediction values
        # todo: (0.2) test different loss functions (see 5.1 https://www.mdpi.com/2072-4292/12/10/1660)
        # todo: (1) increase training samples
        # todo: (2) increase bands / try other input bands (15,29,47)
        # todo: (2.1) use relu for last but one layer to avoid negative values
        # todo: (3) add skip connections
        # todo: (4) add reflection padding for 3d layers
        # todo: add more layers
        # todo: alter kernel sizes / feature maps in 3d layers
        # todo: concat both branches only at the end
        # todo: (use grayscaled msi image)
        # ---
        # todo: atm best config: d23dda2f2837eebaf0d6c84dcfcdac247d86665e OR 9fb725fab3cdddb7b4b5d1a747863dfb6465bc69
        # todo: with skip connection: sharper img
        # todo: negative values possible as long as last layer has a linear activation function https://stats.stackexchange.com/questions/362588/how-can-a-network-with-only-relu-nodes-output-negative-values
        # seed_gen = tf.keras.utils.set_random_seed(42)
        # initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=seed_gen)

        # edge detection
        input2d = Input(shape=(self.tile_size, self.tile_size, 4), name='x')
        kernel = (3, 3)
        leakyRelu = layers.LeakyReLU()
        padded = ReflectionPadding2D(padding=self.padding2d(kernel))(input2d)
        edges1 = layers.Conv2D(3, kernel, padding='valid')(padded)
        edges1 = layers.BatchNormalization()(edges1)
        # edges1 = layers.Activation(leakyRelu)(edges1)
        edges1 = layers.Activation('relu')(edges1)
        padded = ReflectionPadding2D(padding=self.padding2d(kernel))(edges1)
        edges2 = layers.Conv2D(3, kernel, padding='valid')(padded)
        edges2 = layers.BatchNormalization()(edges2)
        # edges2 = layers.Activation(leakyRelu)(edges2)
        edges2 = layers.Activation('relu')(edges2)
        padded = ReflectionPadding2D(padding=self.padding2d(kernel))(edges2)
        edges3 = layers.Conv2D(3, kernel, padding='valid')(padded)
        edges3 = layers.BatchNormalization()(edges3)
        # edges3 = layers.Activation(leakyRelu)(edges3)
        edges3 = layers.Activation('relu')(edges3)

        # main branch
        input3d = Input(shape=(self.tile_size, self.tile_size, self.no_output_bands, 1), name='x1')
        conv1 = layers.Conv3D(64, (9, 9, 7), padding='same', activation=leakyRelu)(input3d)
        merged1 = DILayer()([conv1, edges1])
        merged1 = DILayer()([edges1, merged1])

        skip_connection = layers.Add()([input3d, merged1])

        # todo: restart (currently changing inner kernel sizes)
        # (3, 3, 1) > (1, 1, 1) > (3, 3, 3), 2d layer look more reasonable with (1,1,1) tho
        conv2 = layers.Conv3D(32, (3, 3, 1), padding='same', activation=leakyRelu)(skip_connection)
        merged2 = DILayer()([conv2, edges2])

        skip_connection = layers.Add()([input3d, merged2])

        conv3 = layers.Conv3D(9, (3, 3, 1), padding='same', activation=leakyRelu)(skip_connection)
        # conv3 = layers.Conv3D(9, (3, 3, 1), padding='same', activation='relu')(merged2)
        merged3 = DILayer()([conv3, edges3])

        skip_connection = layers.Add()([input3d, merged3])

        convOut = layers.Conv3D(1, (5, 5, 3), padding='same',
                                activation='linear')(skip_connection)
        y = tf.squeeze(convOut, axis=-1)

        self.model = Model(inputs=[input3d, input2d], outputs=y)


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
        # fcnn with 4d input and 3d output works for 3 input and output bands
        # todo: train with more bands (and more training samples)
        # atm fits with 40 bands but not with 224 --> continue at TestSaPNN


class TestFCNN:
    def __init__(self, tile_size, no_input_bands, no_output_bands):
        self.name = 'TestFCNN'
        self.tile_size = tile_size
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.model = None
        self.create_layers()

    def create_layers(self):
        input3d = Input(shape=(self.tile_size, self.tile_size, self.no_output_bands, 1), name='x1')
        padding3d = (lambda x: (x[0] // 2, x[1] // 2, x[2] // 2))
        kernel = (9, 9, 7)
        pad1 = ReflectionPadding3D(padding=padding3d(kernel))(input3d)
        conv1 = layers.Conv3D(64, kernel, padding='valid',
                              activation='relu')(pad1)
        kernel = (1, 1, 1)
        pad2 = ReflectionPadding3D(padding=padding3d(kernel))(conv1)
        conv2 = layers.Conv3D(32, kernel, padding='valid',
                              activation='relu')(pad2)
        kernel = (1, 1, 1)
        pad3 = ReflectionPadding3D(padding=padding3d(kernel))(conv2)
        conv3 = layers.Conv3D(9, kernel, padding='valid',
                              activation='relu')(pad3)
        kernel = (5, 5, 3)
        pad4 = ReflectionPadding3D(padding=padding3d(kernel))(conv3)
        convOut = layers.Conv3D(1, kernel, padding='valid',
                                activation='linear')(pad4)
        y = tf.squeeze(convOut, axis=-1)

        self.model = Model(inputs=input3d, outputs=y)
