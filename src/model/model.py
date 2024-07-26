import os, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers, regularizers, Input
from matplotlib import pyplot as plt
from image_similarity_measures.quality_metrics import ssim

from .architecture import Masi, ReflectionPadding2D, SaPNN, TestSaPNN, FCNN, TestFCNN, MMSRes
from .load_data import DataGenerator, DuoBranchDataGenerator


# input shape: https://stackoverflow.com/questions/60157742/convolutional-neural-network-cnn-input-shape
# SRCNN: https://github.com/Lornatang/SRCNN-PyTorch/blob/main/model.py 64 - 32 - 1 (no. bands)
# SRCNN: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7115171
# Sentinel CNN: https://github.com/jensleitloff/CNN-Sentinel/blob/master/py/02_train_rgb_finetuning.py
class Model:
    def __init__(self, train_data_dir, tile_size, no_input_bands, no_output_bands, batch_size, kernel_size_list,
                 loss_function, train_epochs, output_dir):
        self.name = None
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
        self.history = None
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
        # architecture = Masi(self.tile_size, self.no_input_bands, self.no_output_bands)

        # SaPNN
        # architecture = SaPNN(self.tile_size, self.no_input_bands, self.no_output_bands)
        # Test
        # architecture = TestSaPNN(self.tile_size, self.no_input_bands, self.no_output_bands)
        # FCNN
        # architecture = FCNN(self.tile_size, self.no_input_bands, self.no_output_bands)
        # architecture = TestFCNN(self.tile_size, self.no_input_bands, self.no_output_bands)
        # MMSRes
        architecture = MMSRes(self.tile_size, self.no_input_bands, self.no_output_bands)

        model = architecture.model
        self.name = architecture.name

        # todo: this already seems to be set by default
        # initializer = initializers.GlorotUniform()
        # for layer in model.layers:
        #     layer.kernel_initializer = initializer

        return model

    def train_test_split(self):
        all_files = os.listdir(self.train_data_dir + 'x/')
        # todo: shuffle? -> in DataGenerator
        self.train_files = all_files[:int(len(all_files) * 0.9)]  # todo: WIP
        self.test_files = all_files[int(len(all_files) * 0.9):]
        # self.train_files = all_files[:5000]  # todo: WIP
        # self.test_files = all_files[5001:6000]
        print('Train data size:', len(self.train_files))
        print('Test data size:', len(self.test_files))

    def set_lr_schedule(self):
        initial_learning_rate = 0.005  # 0.1
        final_learning_rate = 0.0001
        learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / self.train_epochs)
        steps_per_epoch = int(len(self.train_files) / self.batch_size)
        return optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=steps_per_epoch,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

    @staticmethod
    @tf.keras.saving.register_keras_serializable()
    def ms_ssim_l1_loss(y_true, y_pred):
        # loss layer that calculates alpha*(1-MSSSIM)+(1-alpha)*L1 loss
        # https://github.com/NVlabs/PL4NN/blob/master/src/loss.py
        # paper: https://arxiv.org/pdf/1511.08861
        # max_picture_value = 10000
        # alpha = 0.84
        #
        # mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        # msssim_loss = (1 - tf.image.ssim(y_true, y_pred, max_picture_value))
        #
        # loss = (alpha * msssim_loss + (1 - alpha) * mae_loss)
        #
        # return tf.reduce_mean(loss)

        # l1 only
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    def train_model(self):
        train_args = {'data_dir': self.train_data_dir,
                      'data_list': self.train_files,
                      'batch_size': self.batch_size,
                      'output_size': (self.tile_size, self.tile_size),
                      'no_input_bands': self.no_input_bands,
                      'no_output_bands': self.no_output_bands,
                      'shuffle': False}

        test_args = {'data_dir': self.train_data_dir,
                     'data_list': self.test_files,
                     'batch_size': self.batch_size,
                     'output_size': (self.tile_size, self.tile_size),
                     'no_input_bands': self.no_input_bands,
                     'no_output_bands': self.no_output_bands,
                     'shuffle': False}

        # train_generator = DataGenerator(**train_args)
        # test_generator = DataGenerator(**test_args)

        train_generator = DuoBranchDataGenerator(**train_args)
        test_generator = DuoBranchDataGenerator(**test_args)

        self.learning_rate = self.set_lr_schedule()
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        loss = self.ms_ssim_l1_loss  # self.loss_function
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        self.model.summary()

        history = self.model.fit(train_generator, validation_data=test_generator, epochs=self.train_epochs, verbose=1)

        self.history = history.history

        return self.model

    def plot_history(self):
        history = self.history
        acc = history['accuracy']
        val_acc = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, color='royalblue', linewidth=2, label='Training accuracy')
        plt.plot(epochs, val_acc, color='mediumpurple', linewidth=2, linestyle='dashed', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(self.output_dir + 'figures/' + self.name + '_accuracy.png')

        plt.figure()
        plt.plot(epochs, loss, color='firebrick', linewidth=2, label='Training loss')
        plt.plot(epochs, val_loss, color='tomato', linewidth=2, linestyle='dashed', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.output_dir + 'figures/' + self.name + '_loss.png')

        print('Saving model to:', self.output_dir + 'models/' + self.name + '.keras')
        self.model.save(self.output_dir + 'models/' + self.name + '.keras')
