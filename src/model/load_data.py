import numpy as np
import os
# from tensorflow.keras import utils
import tensorflow as tf


class DataGenerator(tf.keras.utils.PyDataset):
    def __init__(self, data_dir, data_list, batch_size, output_size, no_input_bands, no_output_bands, shuffle):
        self.data_dir = data_dir
        self.data_list = data_list
        self.indices = None
        self.batch_size = batch_size
        self.output_size = output_size  # (h, w)
        self.no_input_bands = no_input_bands
        self.no_output_bands = no_output_bands
        self.shuffle = shuffle
        # self.seed = seed
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data_list))  # get data length from directory
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        Common practice: #samples // batch_size (model sees each sample at most once per epoch).
        :return: number of batches per epoch
        """
        return int(len(self.data_list) / self.batch_size)

    def __getitem__(self, idx):
        # (batch_size, w, h, no_input_bands)
        X = np.empty((self.batch_size, *self.output_size, self.no_input_bands))
        Y = np.empty((self.batch_size, *self.output_size, self.no_output_bands))

        # get the indices of the requested batch
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        for i, data_index in enumerate(indices):
            # find img path
            x_path = os.path.join(self.data_dir, 'x', self.data_list[data_index])
            # read img (as np array / df?)
            x_img = np.load(x_path)

            y_path = os.path.join(self.data_dir, 'y', self.data_list[data_index])
            y_img = np.load(y_path)

            # todo: WIP testing with 6 bands
            # x_img = x_img[(50,100,150,225,226,227), :, :]
            # y_img = y_img[(50,100,150), :, :]

            # transpose img as model expects (w, h, no_bands) and img has shape (no_bands, h, w)
            X[i,] = x_img.T
            Y[i,] = y_img.T

        return X, Y


class DuoBranchDataGenerator(DataGenerator):
    def __getitem__(self, idx):
        # (batch_size, w, h, no_input_bands)
        X = np.empty((self.batch_size, *self.output_size, self.no_input_bands))
        X1 = np.empty((self.batch_size, *self.output_size, self.no_output_bands))
        out = np.empty((self.batch_size, *self.output_size, self.no_output_bands))

        # get the indices of the requested batch
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        for i, data_index in enumerate(indices):
            # find img path
            x_path = os.path.join(self.data_dir, 'x', self.data_list[data_index])
            # read img
            x_img = np.load(x_path)

            x1_path = os.path.join(self.data_dir, 'x1', self.data_list[data_index])
            x1_img = np.load(x1_path)

            y_path = os.path.join(self.data_dir, 'y', self.data_list[data_index])
            y_img = np.load(y_path)

            # transpose img as model expects (w, h, no_bands) and img has shape (no_bands, h, w)
            X[i,] = x_img.T
            X1[i,] = x1_img.T
            out[i,] = y_img.T

        print(X.shape, X1.shape, out.shape)

        return {'x': X, 'x1': X1}, out
