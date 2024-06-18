import os, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers, regularizers, Input
from matplotlib import pyplot as plt

from .architecture import Masi, ReflectionPadding2D, SaPnn, TestSaPnn, FCNN, TestFCNN
from .load_data import DataGenerator, DuoBranchDataGenerator


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
        self.learning_rate = None
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

        # Masi
        # model = Masi(self.tile_size, self.no_input_bands, self.no_output_bands).model

        # SaPNN
        # model = SaPnn(self.tile_size, self.no_input_bands, self.no_output_bands).model

        # Test
        # model = TestSaPnn(self.tile_size, self.no_input_bands, self.no_output_bands).model
        # model = FCNN(self.tile_size, self.no_input_bands, self.no_output_bands).model
        model = TestFCNN(self.tile_size, self.no_input_bands, self.no_output_bands).model

        # todo: this already seems to be set by default
        # initializer = initializers.GlorotUniform()
        # for layer in model.layers:
        #     layer.kernel_initializer = initializer

        return model

    def train_test_split(self):
        all_files = os.listdir(self.train_data_dir + 'x/')
        # todo: shuffle? -> in DataGenerator
        self.train_files = all_files[:int(len(all_files) * 0.15)]  # todo: WIP
        self.test_files = all_files[int(len(all_files) * 0.95):]
        print('Train data size:', len(self.train_files))
        print('Test data size:', len(self.test_files))

    def set_lr_schedule(self):
        initial_learning_rate = 0.005
        final_learning_rate = 0.0001
        learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / self.train_epochs)
        steps_per_epoch = int(len(self.train_files) / self.batch_size)
        return optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=steps_per_epoch,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

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

        # train_generator = DuoBranchDataGenerator(self.train_data_dir,
        #                                          data_list=self.train_files,
        #                                          batch_size=self.batch_size,
        #                                          output_size=(self.tile_size, self.tile_size),
        #                                          no_input_bands=self.no_input_bands,
        #                                          no_output_bands=self.no_output_bands,
        #                                          shuffle=False)
        #
        # test_generator = DuoBranchDataGenerator(self.train_data_dir,
        #                                         data_list=self.test_files,
        #                                         batch_size=self.batch_size,
        #                                         output_size=(self.tile_size, self.tile_size),
        #                                         no_input_bands=self.no_input_bands,
        #                                         no_output_bands=self.no_output_bands,
        #                                         shuffle=False)

        self.learning_rate = self.set_lr_schedule()
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        # optimizer = optimizers.Adam(learning_rate=0.001)
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
        # todo: add training time measurement

        print('Saving model to:', self.output_dir + 'models/first_model.keras')
        self.model.save(self.output_dir + 'models/first_model.keras')
        return self.model
