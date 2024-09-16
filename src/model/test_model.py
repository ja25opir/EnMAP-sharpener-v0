import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from image_similarity_measures.quality_metrics import psnr, ssim, sam

from .use_model import prediction as make_prediction


# from src.visualization.helpers import get_bands_from_array
# from src.visualization.plot_raster import plot_3_band_image


def print_layer_names(model):
    print(98 * '-')
    print('Layer names and indices:')
    for i in range(len(model.layers)):
        print(model.layers[i].name, i)
    print(98 * '-')


def plot_band_values(prediction, input_raster, ground_truth, observed_pixel=(15, 15)):
    predicted_values = []
    input_values = []
    ground_truth_values = []
    for band in range(prediction.shape[0]):
        predicted_values.append(prediction[band, observed_pixel[0], observed_pixel[1]])
        input_values.append(input_raster[band, observed_pixel[0], observed_pixel[1]])
        ground_truth_values.append(ground_truth[band, observed_pixel[0], observed_pixel[1]])

    plt.plot(predicted_values, label=f'Prediction for pixel {observed_pixel}', color='green', linewidth=1)
    plt.plot(input_values, label=f'Input for pixel {observed_pixel}', color='orange', linewidth=1, linestyle='dashed')
    plt.plot(ground_truth_values, label=f'Ground truth for pixel {observed_pixel}', color='red', linewidth=1)
    # plt.bar(range(len(ground_truth_values)), ground_truth_values, label=f'Ground truth for pixel {observed_pixel}',
    #         color='green', alpha=0.5)
    plt.legend()
    plt.show()


def get_psnr(prediction, ground_truth, max_p=1):
    mse = np.mean((ground_truth - prediction) ** 2)

    # no noise
    if mse == 0:
        return 100

    return 20 * np.log10(max_p / np.sqrt(mse))


def get_mse(prediction, ground_truth):
    return np.mean((ground_truth - prediction) ** 2)


def evaluate_prediction(prediction, input_x, ground_truth):
    mse_predicted = get_mse(prediction * 1, ground_truth * 1)
    mse_input = get_mse(input_x * 1, ground_truth * 1)

    # get value range from ground truth
    max_pixel_value = ground_truth.max() - ground_truth.min()
    psnr_predicted = psnr(prediction, ground_truth, max_p=max_pixel_value)
    psnr_input = psnr(input_x, ground_truth, max_p=max_pixel_value)

    ssim_predicted = ssim(prediction, ground_truth, max_p=max_pixel_value)
    ssim_input = ssim(input_x, ground_truth, max_p=max_pixel_value)

    # SAM: https://ntrs.nasa.gov/citations/19940012238
    sam_predicted = sam(prediction, ground_truth)
    sam_input = sam(input_x, ground_truth)

    return {'mse_predicted': mse_predicted, 'mse_input': mse_input,
            'psnr_predicted': psnr_predicted, 'psnr_input': psnr_input,
            'ssim_predicted': ssim_predicted, 'ssim_input': ssim_input,
            'sam_predicted': sam_predicted, 'sam_input': sam_input}


def normalize_rasters(prediction, x1, y):
    """Normalize a raster to the range [0, 1] and move the channels to the last dimension."""
    # max_reflectance = 10000
    max_reflectance = 1  # no value normalization
    prediction = (prediction / max_reflectance).T
    x1 = (x1 / max_reflectance).T
    y = (y / max_reflectance).T
    return prediction, x1, y


class Evaluator:
    """
    Class for evaluating a model on given validation scenes.
    """

    def __init__(self, model, validation_data_path, no_output_bands):
        self.model = model
        self.x_data_path = validation_data_path + 'x/'
        self.y_data_path = validation_data_path + 'y/'
        self.validation_scenes = validation_data_path + 'scenes/'
        self.no_output_bands = no_output_bands
        print_layer_names(model)

    def load_data(self, file_name):
        x = np.load(self.x_data_path + file_name)
        y = np.load(self.y_data_path + file_name)
        x = x[(224, 225, 226, 227), :, :]
        x1 = np.load(self.x_data_path + file_name)[:224, :, :]
        # x1 = np.load(self.x_data_path + file_name)[20:60, :, :]  # 40 bands only
        # y = y[20:60, :, :]  # 40 bands only
        return x, x1, y

    def evaluate_on_validation_scene(self, test_file_list):
        """Evaluate the model on a list of test files. Calculate MSE, PSNR, SSIM, SAM and NIQE."""
        evaluations = []
        iterations = len(test_file_list)
        prediction_times = []
        for no in range(iterations):
            x_rast, x1_rast, y_rast = self.load_data(test_file_list[no])
            predicted_rast, prediction_time = make_prediction(x_rast, x1_rast, self.model, self.no_output_bands)
            prediction_times.append(prediction_time)
            # if residual_leaning:
            #     predicted_rast = predicted_rast + x1_rast
            predicted_rast, x1_rast, y_rast = normalize_rasters(predicted_rast, x1_rast, y_rast)
            evaluations.append(evaluate_prediction(predicted_rast, x1_rast, y_rast))
            if (no + 1) % 25 == 0:
                print(f'Evaluated {no + 1} / {iterations} files')

        print(f'Average prediction time: {np.mean(prediction_times):.5f} seconds')
        print('Average evaluation metrics:')
        print(
            f'MSE  p|y: {np.mean([e["mse_predicted"] for e in evaluations]):.2f} | {np.mean([e["mse_input"] for e in evaluations]):.2f}')
        print(
            f'PSNR p|y: {np.mean([e["psnr_predicted"] for e in evaluations]):.2f} | {np.mean([e["psnr_input"] for e in evaluations]):.2f}')
        print(
            f'SSIM p|y: {np.mean([e["ssim_predicted"] for e in evaluations]):.2f} | {np.mean([e["ssim_input"] for e in evaluations]):.2f}')
        print(
            f'SAM  p|y: {np.mean([e["sam_predicted"] for e in evaluations]):.2f} | {np.mean([e["sam_input"] for e in evaluations]):.2f}')

    def evaluate_model(self):
        """Evaluate on validation scenes"""
        print('Evaluating on validation scene from Australia...')
        australia_file = self.validation_scenes + 'Australia.txt'
        with open(australia_file, 'r') as f:
            australia_list = f.read().splitlines()
        self.evaluate_on_validation_scene(australia_list)

        print('Evaluating on validation scene from Namibia...')
        namibia_file = self.validation_scenes + 'Namibia.txt'
        with open(namibia_file, 'r') as f:
            namibia_list = f.read().splitlines()
        self.evaluate_on_validation_scene(namibia_list)

        print('Evaluating on validation scene from Peru...')
        peru_file = self.validation_scenes + 'Peru.txt'
        with open(peru_file, 'r') as f:
            peru_list = f.read().splitlines()
        self.evaluate_on_validation_scene(peru_list)

        print('Evaluating on validation scene from Leipzig...')
        leipzig_file = self.validation_scenes + 'Leipzig.txt'
        with open(leipzig_file, 'r') as f:
            leipzig_list = f.read().splitlines()
        self.evaluate_on_validation_scene(leipzig_list)

    def plot_model_graph(self, output_dir='output/'):
        tf.keras.utils.plot_model(self.model, to_file=output_dir + 'model_graph.png', show_shapes=True)
