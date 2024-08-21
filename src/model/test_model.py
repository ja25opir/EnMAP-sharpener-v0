import os, time

import tensorflow as tf
import numpy as np
import keras.backend as K
from matplotlib import pyplot as plt
from image_similarity_measures.quality_metrics import psnr, ssim, sam


# from src.visualization.helpers import get_bands_from_array
# from src.visualization.plot_raster import plot_3_band_image


def print_layer_names(model):
    print(98 * '-')
    print('Layer names and indices:')
    for i in range(len(model.layers)):
        print(model.layers[i].name, i)
    print(98 * '-')


# def plot_detail_branch(model, x):
#     get_layer_output = (lambda j: K.function(inputs=model.layers[j].input, outputs=model.layers[j].output))
#     padded = get_layer_output(1)(x)
#     first_2d = get_layer_output(2)(padded)
#     first_2d_readable = first_2d[0, :, :, :].T
#     first_2d_batch = get_layer_output(3)(first_2d)
#     first_2d_activation = get_layer_output(4)(first_2d_batch)
#
#     arr = get_bands_from_array(first_2d_activation[0, :, :, :].T, [0, 1, 2])
#     plot_3_band_image(arr, title='First 2d conv + batch + activation')
#
#     padded = get_layer_output(5)(first_2d)
#     second_2d = get_layer_output(6)(padded)
#     second_2d_batch = get_layer_output(8)(second_2d)
#     second_2d_activation = get_layer_output(10)(second_2d_batch)
#     arr = get_bands_from_array(second_2d_activation[0, :, :, :].T, [0, 1, 2])
#     plot_3_band_image(arr, title='Second 2d conv + batch + activation')
#
#     padded = get_layer_output(12)(second_2d)
#     third_2d = get_layer_output(14)(padded)
#     third_2d_batch = get_layer_output(16)(third_2d)
#     third_2d_activation = get_layer_output(18)(third_2d_batch)
#
#     arr = get_bands_from_array(third_2d_activation[0, :, :, :].T, [0, 1, 2])
#     plot_3_band_image(arr, title='Third 2d conv + batch + activation')


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
    mse_predicted = get_mse(prediction * 255, ground_truth * 255)
    mse_input = get_mse(input_x * 255, ground_truth * 255)

    psnr_predicted = psnr(prediction, ground_truth, max_p=1)
    psnr_input = psnr(input_x, ground_truth, max_p=1)

    # todo: SSIM data range [0, 1] or prediction.max() - prediction.min()?
    ssim_predicted = ssim(prediction, ground_truth, max_p=1)
    ssim_input = ssim(input_x, ground_truth, max_p=1)

    # SAM: https://ntrs.nasa.gov/citations/19940012238
    sam_predicted = sam(prediction, ground_truth)
    sam_input = sam(input_x, ground_truth)

    return {'mse_predicted': mse_predicted, 'mse_input': mse_input,
            'psnr_predicted': psnr_predicted, 'psnr_input': psnr_input,
            'ssim_predicted': ssim_predicted, 'ssim_input': ssim_input,
            'sam_predicted': sam_predicted, 'sam_input': sam_input}


def make_prediction(x, x1, model, output_bands):
    x = x.T.reshape(1, 32, 32, 4)
    x1 = x1.T.reshape(1, 32, 32, output_bands, 1)
    return model.predict({'x': x, 'x1': x1}, verbose=0).reshape(32, 32, output_bands).T


def normalize_rasters(prediction, x1, y):
    """Normalize the rasters to the range [0, 1] and move the channels to the last dimension."""
    max_reflectance = 10000
    prediction = (prediction / max_reflectance).T
    x1 = (x1 / max_reflectance).T
    y = (y / max_reflectance).T
    return prediction, x1, y


class Predictor:
    def __init__(self, model, input_data_path, output_data_path, no_output_bands):
        self.model = model
        self.x_data_path = input_data_path
        self.output_data_path = output_data_path
        self.no_output_bands = no_output_bands

    def load_data(self, file_name):
        x = np.load(self.x_data_path + file_name)
        x = x[(224, 225, 226, 227), :, :]
        x1 = np.load(self.x_data_path + file_name)[:224, :, :]
        # x1 = np.load(X_DATA_PATH + file_name)[20:60, :, :]  # 40 bands only
        # y = y[20:60, :, :]  # 40 bands only
        return x, x1

    def make_predictions(self):
        """Make predictions on the input data"""
        i = 1
        files = os.listdir(self.x_data_path)
        no_files = len(files)
        start = time.time()
        for file_name in files:
            x_rast, x1_rast = self.load_data(file_name)
            predicted_rast = make_prediction(x_rast, x1_rast, self.model, self.no_output_bands)
            predicted_rast, x1_rast, _ = normalize_rasters(predicted_rast, x1_rast, x_rast)
            np.save(self.output_data_path + file_name, predicted_rast)
            i += 1
            if i % 25 == 0:
                print(f'Predicted {i} / {no_files} files')
        print(f'Predicted {no_files} files in {time.time() - start} seconds')
        print(f'Average prediction time per file: {(time.time() - start) / no_files} seconds')


class Evaluator:
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
        # x1 = np.load(X_DATA_PATH + file_name)[20:60, :, :]  # 40 bands only
        # y = y[20:60, :, :]  # 40 bands only
        return x, x1, y

    def evaluate_on_validation_scene(self, test_file_list):
        """Evaluate the model on a list of test files. Calculate MSE, PSNR, SSIM, SAM and NIQE."""
        evaluations = []
        iterations = len(test_file_list)
        for no in range(iterations):
            x_rast, x1_rast, y_rast = self.load_data(test_file_list[no])
            predicted_rast = make_prediction(x_rast, x1_rast, self.model, self.no_output_bands)
            # if residual_leaning:
            #     predicted_rast = predicted_rast + x1_rast
            predicted_rast, x1_rast, y_rast = normalize_rasters(predicted_rast, x1_rast, y_rast)
            evaluations.append(evaluate_prediction(predicted_rast, x1_rast, y_rast))
            if (no + 1) % 25 == 0:
                print(f'Evaluated {no + 1} / {iterations} files')

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

# residual_leaning = False
# test_file = '20220627T104548Z_0_0.npy'
# test_file = '20240516T004729Z_16_7.npy'
# test_file = '20240602T155832Z_19_3.npy'
# test_file = '20240611T100311Z_0_0.npy'

# x_raster, x1_raster, y_raster = load_data(test_file)
# predicted_raster = make_prediction(x_raster, x1_raster, sr_model, NO_OUTPUT_BANDS)
#
# plot_bands = [16, 30, 48]
# predicted_rgb = get_bands_from_array(predicted_raster, plot_bands)
# plot_3_band_image(predicted_rgb, title='Predicted Image')
#
# x_rgb = get_bands_from_array(x_raster, [0, 1, 2])
# plot_3_band_image(x_rgb, title='Input Image x (Sentinel)')
# x1_rgb = get_bands_from_array(x1_raster, plot_bands)
# plot_3_band_image(x1_rgb, title='Input Image x1 (EnMAP)')
# y_rgb = get_bands_from_array(y_raster, plot_bands)
# plot_3_band_image(y_rgb, title='Original Image')
#
# if residual_leaning:
#     residual = y_raster - x1_raster
#     residual_rgb = get_bands_from_array(residual, plot_bands)
#     plot_3_band_image(residual_rgb, title='Residual between y and x1')
#
#     residual_added = x1_raster + predicted_raster
#     residual_added_rgb = get_bands_from_array(residual_added, plot_bands)
#     plot_3_band_image(residual_added_rgb, title='Residual added to x1')
#     predicted_raster = residual_added

# plot_detail_branch(sr_model, x_raster.T.reshape(1, 32, 32, 4))

# plot_band_values(predicted_raster, x1_raster, y_raster, observed_pixel=(5, 5))
# plot_band_values(predicted_raster, x1_raster, y_raster, observed_pixel=(10, 10))
# plot_band_values(predicted_raster, x1_raster, y_raster, observed_pixel=(15, 15))
# plot_band_values(predicted_raster, x1_raster, y_raster, observed_pixel=(20, 20))
# plot_band_values(predicted_raster, x1_raster, y_raster, observed_pixel=(25, 25))
