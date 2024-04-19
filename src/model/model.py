import os

from tensorflow.keras import layers, models
from matplotlib import pyplot as plt

from .load_data import DataGenerator


# input shape: https://stackoverflow.com/questions/60157742/convolutional-neural-network-cnn-input-shape
# SRCNN: https://github.com/Lornatang/SRCNN-PyTorch/blob/main/model.py 64 - 32 - 1 (no. bands)
# SRCNN: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7115171
# Sentinel CNN: https://github.com/jensleitloff/CNN-Sentinel/blob/master/py/02_train_rgb_finetuning.py


class Model:
    def __init__(self, train_data_dir, tile_size, no_input_bands, no_output_bands, batch_size, kernel_size_list,
                 loss_function):
        self.train_data_dir = train_data_dir
        self.tile_size = tile_size
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.batch_size = batch_size
        self.kernel_size_list = kernel_size_list
        self.loss_function = loss_function
        self.train_files = None
        self.test_files = None
        self.model = self.define_model()
        self.train_test_split()

    def define_model(self):
        # padding = same (output size = input size) --> rethink this
        # activation function relu, relu, linear (Masi) --> rethink this
        # layers described in Masi p.4 (2.2) 64 - 32 - 3 (no. bands) kernels: 9x9, 1x1 (3x3), 5x5
        model = models.Sequential()
        model.add(layers.Conv2D(512,
                                self.kernel_size_list[0],
                                activation='relu',
                                input_shape=(self.tile_size, self.tile_size, self.no_input_bands),
                                padding='same'))
        model.add(layers.Conv2D(256,
                                self.kernel_size_list[1],
                                activation='relu',
                                padding='same'))
        model.add(layers.Conv2D(self.no_output_bands,
                                self.kernel_size_list[2],
                                activation='linear',
                                padding='same'))
        model.summary()
        return model

    def train_test_split(self):
        all_files = os.listdir(self.train_data_dir + 'x/')
        # todo: shuffle?
        self.train_files = all_files[:int(len(all_files) * 0.8)]
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

        self.model.compile(optimizer='adam', loss=self.loss_function, metrics=['accuracy'])

        history = self.model.fit(train_generator, validation_data=test_generator, epochs=10, verbose=1)

        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        # plt.show()
        plt.savefig('../../output/figures/first_model_accuracy.png')
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.savefig('../../output/figures/first_model_loss.png')

        self.model.save('../../output/models/first_model.keras')
