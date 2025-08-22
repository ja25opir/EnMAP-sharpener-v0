import tensorflow as tf
from tensorflow.keras import layers, Input, Model


@tf.keras.utils.register_keras_serializable()
def ms_ssim_l1_loss(y_true, y_pred):
    # alpha * (1 - MS_SSIM) + (1 - alpha) * L1_loss
    # source: https://arxiv.org/pdf/1511.08861
    max_raster_value = 10000
    alpha = 0.84

    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))  # == mean absolute error
    ms_ssim_loss = (1 - tf.image.ssim(y_true, y_pred, max_raster_value))

    loss = (alpha * ms_ssim_loss + (1 - alpha) * l1)

    return tf.reduce_mean(loss)


@tf.keras.utils.register_keras_serializable()
def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


@tf.keras.utils.register_keras_serializable()
def residual_loss(y_true, y_pred):
    # source: https://openaccess.thecvf.com/content_cvpr_2016/papers/Kim_Accurate_Image_Super-Resolution_CVPR_2016_paper.pdf
    return 1 / 2 * tf.square(tf.abs(y_true - y_pred))


@tf.keras.utils.register_keras_serializable()
def ssim(y_true, y_pred):
    max_raster_value = 10000
    return tf.image.ssim(y_true, y_pred, max_raster_value)


@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


@tf.keras.utils.register_keras_serializable()
def variance(y_true, y_pred):
    return tf.math.reduce_variance(y_true - y_pred)


@tf.keras.utils.register_keras_serializable()
def psnr(y_true, y_pred):
    max_raster_value = 10000
    return tf.image.psnr(y_true, y_pred, max_raster_value)


class ReflectionPadding2D(layers.Layer):
    """
    2D reflection padding layer.
    """
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


class ReflectionPadding3D(layers.Layer):
    """
    3D reflection padding layer.
    """
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


class DILayer(layers.Layer):
    """
    Detail injection layer that stacks feature maps of the main branch with feature maps of the detail detection branch.
    """

    def __init__(self, kernel_size=(3, 3), **kwargs):
        super(DILayer, self).__init__(**kwargs)
        self.kernel = kernel_size
        self.x_shape = None

    def build(self, input_shape):
        self.x_shape = input_shape[0]

    def call(self, inputs):
        x, edges = inputs
        merged = None

        # stack edges feature maps to input feature maps
        for band_no in range(self.x_shape[-2]):
            x_band = x[:, :, :, band_no, :]
            x_stacked = tf.concat([x_band, edges], axis=-1)
            x_stacked = tf.expand_dims(x_stacked, axis=-2)
            if merged is None:
                merged = x_stacked
            else:
                merged = tf.concat([merged, x_stacked], axis=-2)

        return merged


class SupErMAPnet:
    """
    SuperMAPnet model architecture that sharpens hyperspectral EnMAP images using auxiliary multispectral Sentinel-2 data.
    """

    def __init__(self, tile_size, no_input_bands, no_output_bands, kernels_mb, kernels_db, filters_mb, filters_db):
        self.name = 'supErMAPnet'
        self.tile_size = tile_size
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.kernels_mb = kernels_mb
        self.kernels_db = kernels_db
        self.filters_mb = filters_mb
        self.filters_db = filters_db
        self.padding2d = (lambda x: (x[0] // 2, x[1] // 2))
        self.model = None
        self.create_layers()

    def create_layers(self):
        """
        Create the layers of the model using a detail detection branch and a main branch.
        :return: None
        """

        # detail detection
        input2d = Input(shape=(self.tile_size, self.tile_size, 4), name='x')
        leaky_relu = layers.LeakyReLU()
        padded = ReflectionPadding2D(padding=self.padding2d(self.kernels_db[0]))(input2d)
        edges1 = layers.Conv2D(self.filters_db[0], self.kernels_db[0], padding='valid')(padded)
        edges1 = layers.BatchNormalization(momentum=0.9, epsilon=0.01)(edges1)
        edges1 = layers.Activation(leaky_relu)(edges1)
        padded = ReflectionPadding2D(padding=self.padding2d(self.kernels_db[1]))(edges1)
        edges2 = layers.Conv2D(self.filters_db[1], self.kernels_db[1], padding='valid')(padded)
        edges2 = layers.BatchNormalization(momentum=0.9, epsilon=0.01)(edges2)
        edges2 = layers.Activation(leaky_relu)(edges2)
        padded = ReflectionPadding2D(padding=self.padding2d(self.kernels_db[2]))(edges2)
        edges3 = layers.Conv2D(self.filters_db[2], self.kernels_db[2], padding='valid')(padded)
        edges3 = layers.BatchNormalization(momentum=0.9, epsilon=0.01)(edges3)
        edges3 = layers.Activation(leaky_relu)(edges3)

        # main branch
        input3d = Input(shape=(self.tile_size, self.tile_size, self.no_output_bands, 1), name='x1')
        conv1 = layers.Conv3D(self.filters_mb[0], self.kernels_mb[0], padding='same', activation=leaky_relu)(input3d)
        merged1 = DILayer()([conv1, edges1])

        conv2 = layers.Conv3D(self.filters_mb[1], self.kernels_mb[1], padding='same', activation=leaky_relu)(merged1)
        merged2 = DILayer()([conv2, edges2])

        conv3 = layers.Conv3D(self.filters_mb[2], self.kernels_mb[2], padding='same', activation=leaky_relu)(merged2)
        merged3 = DILayer()([conv3, edges3])

        conv_out = layers.Conv3D(1, self.kernels_mb[3], padding='same',
                                activation='linear')(merged3)

        skip_connection = layers.Add()([input3d, conv_out])

        y = tf.squeeze(skip_connection, axis=-1)

        self.model = Model(inputs=[input3d, input2d], outputs=y)
