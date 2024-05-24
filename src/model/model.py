import os, random

import tensorflow as tf
from tensorflow.keras import layers, models, initializers, regularizers
from tensorflow.keras.layers import Layer, Input
from matplotlib import pyplot as plt

from .load_data import DataGenerator


class SymmetricPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(SymmetricPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1],
                input_shape[3])

    def __call__(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor,
                      tf.constant([[0, 0],
                                   [padding_height, padding_height],
                                   [padding_width, padding_width],
                                   [0, 0]]),
                      'SYMMETRIC')


# input shape: https://stackoverflow.com/questions/60157742/convolutional-neural-network-cnn-input-shape
# SRCNN: https://github.com/Lornatang/SRCNN-PyTorch/blob/main/model.py 64 - 32 - 1 (no. bands)
# SRCNN: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7115171
# Sentinel CNN: https://github.com/jensleitloff/CNN-Sentinel/blob/master/py/02_train_rgb_finetuning.py
class Model:
    def __init__(self, train_data_dir, tile_size, no_input_bands, no_output_bands, batch_size, kernel_size_list,
                 loss_function, train_epochs, output_dir):
        self.train_data_dir = train_data_dir
        self.tile_size = tile_size
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.batch_size = batch_size
        self.kernel_size_list = kernel_size_list
        self.loss_function = loss_function
        self.train_epochs = train_epochs
        self.output_dir = output_dir
        self.train_files = None
        self.test_files = None
        self.model = self.define_model()
        self.train_test_split()

    def define_model(self):
        # padding = same (output size = input size) --> rethink this
        # activation function relu, relu, linear (Masi) --> rethink this
        # maybe leaky relu vs vanishing gradients
        # layers described in Masi p.4 (2.2) 64 - 32 - 3 (no. bands) kernels: 9x9, 1x1 (3x3), 5x5
        # todo: add padding to tiles (overlap in windows to avoid edge effects caused by zero padding)
        # maybe use 3d kernels as spectral bands may be "connected" (but maybe sentinel layers need to be sorted in then)
        # maybe add more sentinel layers as duplicates (bias) to increase the importance of these layers
        # padding: https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
        # stride: https://tcnguyen.github.io/neuralnetworks/cnn_tensorflow.html
        # padding: https://hidayatullahhaider.medium.com/a-simple-definition-of-overlap-term-in-cnn-f331f6ef3031
        # padding: https://openreview.net/pdf?id=M4qXqdw3xC#:~:text=Recent%20studies%20have%20shown%20that,of%20padding%20precludes%20position%20encoding
        model = models.Sequential()
        # experimental
        # model.add(layers.Conv2D(1024,
        #                         self.kernel_size_list[0],
        #                         activation='relu',
        #                         input_shape=(self.tile_size, self.tile_size, self.no_input_bands),
        #                         padding='same'))
        # model.add(layers.Conv2D(512,
        #                         self.kernel_size_list[1],
        #                         activation='relu',
        #                         padding='same'))
        # model.add(layers.Conv2D(512,
        #                         self.kernel_size_list[1],
        #                         activation='relu',
        #                         padding='same'))
        # model.add(layers.Conv2D(512,
        #                         self.kernel_size_list[1],
        #                         activation='relu',
        #                         padding='same'))
        # model.add(layers.Conv2D(512,
        #                         self.kernel_size_list[1],
        #                         activation='relu',
        #                         padding='same'))
        # model.add(layers.Conv2D(512,
        #                         self.kernel_size_list[1],
        #                         activation='relu',
        #                         padding='same'))
        # model.add(layers.Conv2D(256,
        #                         self.kernel_size_list[2],
        #                         activation='relu',
        #                         padding='same'))
        # relu in last layer significantly increased accuracy but worsened loss (stuck after one epoch)
        #  --> prediction with this model is all 0 this is why the accuracy is 1/3 (many 0s in input data)
        # model.add(layers.Conv2D(self.no_output_bands,
        #                         self.kernel_size_list[2],
        #                         activation='linear',
        #                         padding='same'))

        # very deep network:
        # one hidden layer per output band
        input_shape = (self.tile_size, self.tile_size, self.no_input_bands)
        model.add(Input(shape=input_shape))
        model.add(SymmetricPadding2D(padding=(4, 4)))
        model.add(layers.Conv2D(64,
                                self.kernel_size_list[0],
                                # activation=tf.keras.layers.LeakyReLU(),
                                activation='relu',
                                kernel_regularizer=regularizers.l1(0.01),
                                padding='valid'))
        for i in range(10):
            model.add(SymmetricPadding2D(padding=(1, 1)))
            model.add(layers.Conv2D(64,
                                    self.kernel_size_list[1],
                                    # activation=tf.keras.layers.LeakyReLU(),
                                    activation='relu',
                                    kernel_regularizer=regularizers.l1(0.01),
                                    padding='valid'))
        model.add(SymmetricPadding2D(padding=(2, 2)))
        model.add(layers.Conv2D(self.no_output_bands,
                                self.kernel_size_list[2],
                                activation='linear',
                                padding='valid'))

        # Masi
        # model.add(layers.Conv2D(64,
        #                         (9,9),
        #                         activation='relu',
        #                         input_shape=(self.tile_size, self.tile_size, self.no_input_bands),
        #                         padding='same'))
        # model.add(layers.Conv2D(32,
        #                         (5,5),
        #                         activation='relu',
        #                         padding='same'))
        # model.add(layers.Conv2D(self.no_output_bands,
        #                         (5,5),
        #                         activation='linear',
        #                         padding='same'))

        # todo: this already seems to be set by default
        # initializer = initializers.GlorotUniform()
        # for layer in model.layers:
        #     layer.kernel_initializer = initializer

        # model.summary()
        return model

    def train_test_split(self):
        all_files = os.listdir(self.train_data_dir + 'x/')
        # todo: shuffle? -> in DataGenerator
        self.train_files = all_files[:int(len(all_files) * 0.8)]  # todo: WIP
        self.test_files = all_files[int(len(all_files) * 0.8):]
        print('Train data size:', len(self.train_files))
        print('Test data size:', len(self.test_files))

    def train_model(self):
        train_generator = DataGenerator(self.train_data_dir,
                                        data_list=self.train_files,
                                        batch_size=self.batch_size,
                                        output_size=(self.tile_size, self.tile_size),
                                        no_input_bands=self.no_input_bands,
                                        no_output_bands=self.no_output_bands,
                                        shuffle=False)

        test_generator = DataGenerator(self.train_data_dir,
                                       data_list=self.test_files,
                                       batch_size=self.batch_size,
                                       output_size=(self.tile_size, self.tile_size),
                                       no_input_bands=self.no_input_bands,
                                       no_output_bands=self.no_output_bands,
                                       shuffle=False)

        optimizer = tf.keras.optimizers.Adam(lr=0.00001)
        self.model.compile(optimizer=optimizer, loss=self.loss_function, metrics=['accuracy'])
        self.model.summary()

        history = self.model.fit(train_generator, validation_data=test_generator, epochs=self.train_epochs, verbose=1)

        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        # plt.show()
        plt.savefig(self.output_dir + 'models/first_model_accuracy.png')
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.savefig(self.output_dir + 'models/first_model_loss.png')

        print('Saving model to:', self.output_dir + 'models/first_model.keras')
        self.model.save(self.output_dir + 'models/first_model.keras')
        return self.model
