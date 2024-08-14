import os, random, time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers, regularizers, Input
from matplotlib import pyplot as plt

from .architecture import Masi, ReflectionPadding2D, SaPNN, TestSaPNN, FCNN, TestFCNN, MMSRes, ms_ssim_l1_loss, \
    residual_loss, ssim, mse, variance, psnr
from .load_data import DuoBranchDataGenerator


# input shape: https://stackoverflow.com/questions/60157742/convolutional-neural-network-cnn-input-shape
# SRCNN: https://github.com/Lornatang/SRCNN-PyTorch/blob/main/model.py 64 - 32 - 1 (no. bands)
# SRCNN: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7115171
# Sentinel CNN: https://github.com/jensleitloff/CNN-Sentinel/blob/master/py/02_train_rgb_finetuning.py
class Model:
    def __init__(self, train_data_dir, tile_size, no_input_bands, no_output_bands, batch_size,
                 loss_function, train_epochs, output_dir):
        self.model = None
        self.name = None
        self.train_data_dir = train_data_dir
        self.tile_size = tile_size
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.train_epochs = train_epochs
        self.output_dir = output_dir
        self.train_files = None
        self.test_files = None
        self.learning_rate = None
        self.history = None
        self.train_test_split()

    def define_model(self, hyperparameters=None):
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
        architecture = MMSRes(self.tile_size, self.no_input_bands, self.no_output_bands,
                              kernels_mb=hyperparameters['k_mb'],
                              kernels_db=hyperparameters['k_db'],
                              filters_mb=hyperparameters['f_mb'],
                              filters_db=hyperparameters['f_db'])

        self.name = architecture.name
        self.model = architecture.model

    def train_test_split(self):
        all_files = os.listdir(self.train_data_dir + 'x/')
        all_files = random.sample(all_files, len(all_files))
        train_ratio = 0.01
        self.train_files = all_files[:int(len(all_files) * train_ratio)]
        self.test_files = all_files[int(len(all_files) * 0.995):]
        print('Train data size:', len(self.train_files))
        print('Test data size:', len(self.test_files))

    def set_lr_schedule(self):
        initial_learning_rate = 0.0001  # training 224 bands # 0.001 hyperparams
        final_learning_rate = 0.00001
        learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / self.train_epochs)
        steps_per_epoch = int(len(self.train_files) / self.batch_size)
        return optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=steps_per_epoch,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

    def train_model(self,
                    kernels_mb: list[tuple[int, int, int]],
                    kernels_db: list[tuple[int, int]],
                    filters_mb: list[int],
                    filters_db: list[int]) -> None:
        """
        Train the model with the given hyperparameters.
        :param kernels_mb: list of kernel sizes for the main branch
        :param kernels_db: list of kernel sizes for the detail branch
        :param filters_mb: list of number of filters for each layer in the main branch
        :param filters_db: list of number of filters for each layer in the detail branch
        :return: None
        """
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

        train_generator = DuoBranchDataGenerator(**train_args)
        test_generator = DuoBranchDataGenerator(**test_args)

        self.learning_rate = self.set_lr_schedule()
        loss = ms_ssim_l1_loss  # self.loss_function

        self.define_model(
            hyperparameters={'k_mb': kernels_mb, 'k_db': kernels_db, 'f_mb': filters_mb, 'f_db': filters_db})
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[ssim, psnr, mse, variance])
        self.model.summary()
        stop_nan = tf.keras.callbacks.TerminateOnNaN()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          min_delta=1,
                                                          patience=3,
                                                          restore_best_weights=True)
        history = None
        attempts = 3
        # retry twice if loss is nan
        while history is None or np.isnan(history.history['loss'][-1]) and attempts > 0:
            history = self.model.fit(train_generator,
                                     validation_data=test_generator,
                                     epochs=self.train_epochs,
                                     verbose=1,
                                     callbacks=[stop_nan, early_stopping])
            attempts -= 1

        self.history = history.history

    def fit_hyperparameter(self) -> None:
        """
        Train model with different hyperparameters, compare and save best model.
        :return: trained model with best accuracy
        """
        kernel_sizes_db = [
            # [(3, 3), (3, 3), (3, 3)],
            # [(7, 7), (7, 7), (7, 7)],
            # [(9, 9), (3, 3), (5, 5)],
            [(9, 9), (5, 5), (3, 3)],
        ]
        kernel_sizes_mb = [
            # [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
            # [(7, 7, 7), (7, 7, 7), (7, 7, 7), (7, 7, 7)],
            # [(9, 9, 7), (1, 1, 1), (1, 1, 1), (5, 5, 3)],
            # [(9, 9, 7), (3, 3, 1), (3, 3, 1), (5, 5, 3)],
            # [(9, 9, 7), (3, 3, 3), (3, 3, 3), (5, 5, 3)],
            [(9, 9, 7), (3, 3, 6), (3, 3, 6), (5, 5, 3)]
        ]

        filters_mb = [
            [64, 64, 64],
            # [64, 32, 9]
        ]
        filters_db = [
            # [64, 64, 64],
            [64, 32, 9],
            # [9, 6, 3],
            # [3, 3, 3]
        ]

        best_ssim_psnr = 0
        best_kernels = None
        best_filters = None
        history_list = []
        start = time.time()
        for k_mb in kernel_sizes_mb:
            for k_db in kernel_sizes_db:
                for f_mb in filters_mb:
                    for f_db in filters_db:
                        self.model = None

                        print('Main branch kernels: ', k_mb)
                        print('Detail branch kernels: ', k_db)
                        print('Main branch filters: ', f_mb)
                        print('Detail branch filters: ', f_db)

                        self.train_model(kernels_mb=k_mb, kernels_db=k_db, filters_mb=f_mb, filters_db=f_db)

                        # history_list.append({'k_mb': k_mb, 'k_db': k_db, 'hist': self.history})
                        history_list.append({'f_mb': f_mb, 'f_db': f_db, 'hist': self.history})

                        ssim_psnr = self.history['val_ssim'][-1] + self.history['val_psnr'][-1] / 100

                        # save model if better than previous (metric: ssim + psnr / 100)
                        if ssim_psnr > best_ssim_psnr:
                            print('New best model found! Saving...')
                            best_ssim_psnr = ssim_psnr
                            best_kernels = {'k_mb': k_mb, 'k_db': k_db}
                            best_filters = {'f_mb': f_mb, 'f_db': f_db}
                            self.model.save(self.output_dir + 'models/' + self.name + '.keras')
                            print('Saved model:', self.name)

                        print('-' * 98)

                        # clear sequential model graph and delete model to avoid clutter from old models and free memory
                        # tf.keras.backend.clear_session()

        # save history list
        with open(self.output_dir + 'models/' + self.name + '_hyperparam_history.txt', 'w') as f:
            f.write(str(history_list))

        print("Hyperparameter search finished!")
        print("Best Kernels: \n", best_kernels)
        print("Best Filters: \n", best_filters)
        end = time.time()
        print("---HyperparameterSearch---Elapsed time: %.2fs seconds ---" % (end - start))

    def plot_history(self):
        history = self.history
        train_ssim = history['ssim']
        val_ssim = history['val_ssim']
        train_psnr = history['psnr']
        val_psnr = history['val_psnr']
        loss = history['loss']
        val_loss = history['val_loss']

        epochs = range(1, len(train_ssim) + 1)

        fig, ax = plt.subplots()
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))

        plt.plot(epochs, train_ssim, color='tomato', linewidth=1, label='Training SSIM')
        plt.plot(epochs, val_ssim, color='mediumpurple', linewidth=1, linestyle='dashed', label='Validation SSIM')
        plt.title('Training and validation SSIM')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.legend()
        plt.savefig(self.output_dir + 'figures/' + self.name + '_SSIM.png')

        plt.plot(epochs, train_psnr, color='tomato', linewidth=1, label='Training PSNR')
        plt.plot(epochs, val_psnr, color='mediumpurple', linewidth=1, linestyle='dashed', label='Validation PSNR')
        plt.title('Training and validation PSNR')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.legend()
        plt.savefig(self.output_dir + 'figures/' + self.name + '_PSNR.png')

        plt.figure()
        plt.plot(epochs, loss, color='firebrick', linewidth=1, label='Training loss')
        plt.plot(epochs, val_loss, color='royalblue', linewidth=1, linestyle='dashed', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.output_dir + 'figures/' + self.name + '_loss.png')

        print('Saving model to:', self.output_dir + 'models/' + self.name + '.keras')
        self.model.save(self.output_dir + 'models/' + self.name + '.keras')
