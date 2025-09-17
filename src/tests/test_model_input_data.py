import os
import numpy as np
import unittest
import matplotlib.pyplot as plt
from itertools import islice

DEFAULT_DIR = os.getcwd() + '/data/preprocessing/model_input/'


class TestModelInputData(unittest.TestCase):
    def setUp(self, data_dir=DEFAULT_DIR):
        self.data_dir_x = data_dir + 'x'
        self.data_dir_y = data_dir + 'y'

    # currently not used as this a) loads all filenames in mem and b) could be done iteratively as the model data loader loads sorted batches of files
    def _test_file_pairs(self):
        spectral_mat_x = [x for x in os.listdir(self.data_dir_x) if x.endswith('.npy')]
        spectral_mat_y = [y for y in os.listdir(self.data_dir_y) if y.endswith('.npy')]

        # check if each file in x has a corresponding file in y
        for file_x in spectral_mat_x:
            self.assertIn(file_x, spectral_mat_y, f"Missing corresponding file for {file_x} in y directory.")
        # check if each file in y has a corresponding file in x
        for file_y in spectral_mat_y:
            self.assertIn(file_y, spectral_mat_x, f"Missing corresponding file for {file_y} in x directory.")

    def check_data_shape(self, data_dir, shape):
        for entry in os.scandir(data_dir):
            if entry.is_file() and entry.name.endswith('.npy'):
                data = np.load(os.path.join(data_dir, entry))
                self.assertTrue(data.shape == shape,
                                f"File {entry} has a dimension of {data.shape}, expected {shape}.")

    def test_data_shape(self):
        self.check_data_shape(self.data_dir_x, (228, 32, 32))
        self.check_data_shape(self.data_dir_y, (224, 32, 32))

    def check_value_range(self, data_dir):
        for entry in os.scandir(data_dir):
            if entry.is_file() and entry.name.endswith('.npy'):
                data = np.load(os.path.join(data_dir, entry))
                self.assertTrue(np.all(np.isfinite(data)), f"File {entry} contains NaN or infinite values.")
                self.assertTrue(data.dtype == np.float32,
                                f"File {entry} has an unsupported data type: {data.dtype}. Expected np.float32.")
                self.assertTrue(np.all((data >= 0) & (data <= 10000)),
                                f"Data in {entry} is not in the range [0, 10000].")

    def test_value_range(self):
        self.check_value_range(self.data_dir_x)
        self.check_value_range(self.data_dir_y)

    def plot_spectral_signatures(self, data_dir):
        spectral_mat = [mat for mat in os.listdir(data_dir) if mat.endswith('.npy')]

        for file in spectral_mat:
            data = np.load(os.path.join(data_dir, file))
            num_bands, height, width = data.shape
            spectra = data.reshape(num_bands, -1).T

            plt.figure(figsize=(10, 6))
            for spectrum in spectra:
                plt.plot(range(1, num_bands + 1), spectrum, color='gray', alpha=0.3, linewidth=0.8)

            # Mean spectrum for emphasis
            mean_spectrum = np.nanmean(spectra, axis=0)
            plt.plot(range(1, num_bands + 1), mean_spectrum, color='red', linewidth=2, label='Mean Spectrum')

            plt.title("Spectral Signatures in Auwald Window 10m")
            plt.xlabel("Band Number")
            plt.ylabel("Reflectance / DN Value")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()

        self.assertTrue(len(spectral_mat) > 0, "No spectral signature files found in the directory.")

    def _test_plot_spectral_signatures(self):
        self.plot_spectral_signatures(self.data_dir_x)
        self.plot_spectral_signatures(self.data_dir_y)


if __name__ == '__main__':
    unittest.main()
