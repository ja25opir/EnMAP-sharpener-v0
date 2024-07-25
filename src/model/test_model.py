import os, random
import rasterio
import tensorflow as tf
import numpy as np
import keras.backend as K
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
from image_similarity_measures.quality_metrics import psnr, ssim, sam

from src.model.architecture import ReflectionPadding2D, ReflectionPadding3D, SFTLayer, DILayer
from src.visualization.helpers import get_bands_from_array
from src.visualization.plot_raster import plot_3_band_image


def plot_detail_branch(model):
    get_layer_output = (lambda j: K.function(inputs=model.layers[j].input, outputs=model.layers[j].output))
    padded = get_layer_output(1)(x)  # TODO: is this the wrong input lol?
    first_2d = get_layer_output(2)(padded)
    first_2d_readable = first_2d[0, :, :, :].T
    first_2d_batch = get_layer_output(3)(first_2d)
    first_2d_activation = get_layer_output(4)(first_2d_batch)

    arr = get_bands_from_array(first_2d_activation[0, :, :, :].T, [0, 1,
                                                                   2])  # todo: only extract 3 feat maps with 2d convs and inject (merged = 3 * 64 feature maps for first layer)
    plot_3_band_image(arr, title='First 2d conv + batch + activation')

    padded = get_layer_output(5)(first_2d)
    second_2d = get_layer_output(6)(padded)
    second_2d_batch = get_layer_output(8)(second_2d)
    second_2d_activation = get_layer_output(10)(second_2d_batch)
    arr = get_bands_from_array(second_2d_activation[0, :, :, :].T, [0, 1, 2])
    plot_3_band_image(arr, title='Second 2d conv + batch + activation')

    padded = get_layer_output(12)(second_2d)
    third_2d = get_layer_output(14)(padded)
    third_2d_batch = get_layer_output(16)(third_2d)
    third_2d_activation = get_layer_output(18)(third_2d_batch)

    arr = get_bands_from_array(third_2d_activation[0, :, :, :].T, [0, 1, 2])
    plot_3_band_image(arr, title='Third 2d conv + batch + activation')


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


CUSTOM_LAYERS = {'ReflectionPadding2D': ReflectionPadding2D,
                 'ReflectionPadding3D': ReflectionPadding3D,
                 'SFTLayer': SFTLayer,
                 'DILayer': DILayer}

X_DATA_PATH = os.getcwd() + '/../../data/preprocessing/model_input/x/'
X1_DATA_PATH = os.getcwd() + '/../../data/preprocessing/model_input/x1/'
Y_DATA_PATH = os.getcwd() + '/../../data/preprocessing/model_input/y/'
MODEL_PATH = os.getcwd() + '/../../output/models/'

x_data = os.listdir(X_DATA_PATH)
y_data = os.listdir(Y_DATA_PATH)

# get random file from data
# test_file = random.choice(y_data)
test_file = '20220627T104548Z_0_0.npy'

x_raster = np.load(X_DATA_PATH + test_file)
x1_raster = np.load(X1_DATA_PATH + test_file)
y_raster = np.load(Y_DATA_PATH + test_file)

sr_model = tf.keras.models.load_model(MODEL_PATH + 'MMSRes.keras', custom_objects=CUSTOM_LAYERS)

print(sr_model.summary())

NO_OUTPUT_BANDS = 20
x_raster = x_raster[(224, 225, 226, 227), :, :]
# x1_raster = x1_raster[(15, 29, 47, 71), :, :]  # 4 bands only
x1_raster = x1_raster[20:40, :, :]  # 20 bands only
y_raster = y_raster[20:40, :, :]  # 20 bands only
x = x_raster.T.reshape(1, 32, 32, 4)
x1 = x1_raster.T.reshape(1, 32, 32, NO_OUTPUT_BANDS, 1)
predicted_raster = sr_model.predict({'x': x, 'x1': x1}).reshape(32, 32, NO_OUTPUT_BANDS).T

bands = [0, 1, 2]
predicted_rgb = get_bands_from_array(predicted_raster, bands)
plot_3_band_image(predicted_rgb, title='Predicted Image')

x_rgb = get_bands_from_array(x_raster, bands)
plot_3_band_image(x_rgb, title='Input Image x (Sentinel)')
x_rgb = get_bands_from_array(x1_raster, bands)
plot_3_band_image(x_rgb, title='Input Image x1 (EnMAP)')
y_rgb = get_bands_from_array(y_raster, bands)
plot_3_band_image(y_rgb, title='Original Image')

# tf.keras.utils.plot_model(model, to_file='model_graph.png', show_shapes=True)

print('Layer names and indices')
for i in range(len(sr_model.layers)):
    print(sr_model.layers[i].name, i)
print('----------')

# plot_detail_branch(sr_model)

# plot_band_values(predicted_raster, x1_raster, y_raster[20:40], observed_pixel=(5, 5))
# plot_band_values(predicted_raster, x1_raster, y_raster[20:40], observed_pixel=(10, 10))
# plot_band_values(predicted_raster, x1_raster, y_raster[20:40], observed_pixel=(15, 15))
# plot_band_values(predicted_raster, x1_raster, y_raster[20:40], observed_pixel=(20, 20))
# plot_band_values(predicted_raster, x1_raster, y_raster[20:40], observed_pixel=(25, 25))

# shift data range to [0, 1] and move channels to last dimension
MAX_REFLECTANCE = 10000
predicted_raster = (predicted_raster / MAX_REFLECTANCE).T
x1_raster = (x1_raster / MAX_REFLECTANCE).T
y_raster = (y_raster / MAX_REFLECTANCE).T

# todo: evaluation metrics: MSE, PSNR, SSIM, SAM
mse_predicted = get_mse(predicted_raster * 255, y_raster * 255) # todo: which factor?
mse_input = get_mse(x1_raster * 255, y_raster * 255)
print(f'MSE: {mse_predicted:.2f} (predicted) vs. {mse_input:.2f} (input) | 0 is perfect similarity')

psnr_predicted = get_psnr(predicted_raster, y_raster)
psnr_input = get_psnr(x1_raster, y_raster)
print(f'PSNR: {psnr_predicted:.2f} (predicted) vs. {psnr_input:.2f} (input) | 100 is perfect similarity')
psnr_predicted = psnr(predicted_raster, y_raster, max_p=1)
psnr_input = psnr(x1_raster, y_raster, max_p=1)
print(f'PSNR: {psnr_predicted:.2f} (predicted) vs. {psnr_input:.2f} (input) | 100 is perfect similarity')

# todo: SSIM data range [0, 1] or prediction.max() - prediction.min()?
ssim_predicted = structural_similarity(predicted_raster, y_raster, multichannel=True,
                                       data_range=1)
ssim_input = structural_similarity(x1_raster, y_raster, multichannel=True,
                                   data_range=1)
print(f'SSIM: {ssim_predicted:.2f} (predicted) vs. {ssim_input:.2f} (input) | 1.0 is perfect similarity')
ssim_predicted = ssim(predicted_raster, y_raster, max_p=1)
ssim_input = ssim(x1_raster, y_raster, max_p=1)
print(f'SSIM: {ssim_predicted:.2f} (predicted) vs. {ssim_input:.2f} (input) | 1.0 is perfect similarity')

# SAM: https://ntrs.nasa.gov/citations/19940012238
sam_predicted = sam(predicted_raster, y_raster)
sam_input = sam(x1_raster, y_raster)
print(f'SAM: {sam_predicted:.2f} (predicted) vs. {sam_input:.2f} (input) | 0 is perfect similarity')