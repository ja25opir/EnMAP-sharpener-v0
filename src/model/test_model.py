import os, random
import rasterio
import tensorflow as tf
import numpy as np
import keras.backend as K

from src.model.architecture import ReflectionPadding2D, ReflectionPadding3D, SFTLayer, DILayer
from src.visualization.helpers import get_bands_from_array
from src.visualization.plot_raster import plot_3_band_image

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

model = tf.keras.models.load_model(MODEL_PATH + 'MMSRes.keras', custom_objects=CUSTOM_LAYERS)

print(model.summary())

NO_OUTPUT_BANDS = 80
x_raster = x_raster[(224, 225, 226, 227), :, :]
# x1_raster = x1_raster[(15, 29, 47, 71), :, :]  # 4 bands only
x1_raster = x1_raster[20:40, :, :]  # 20 bands only
x = x_raster.T.reshape(1, 32, 32, 4)
x1 = x1_raster.T.reshape(1, 32, 32, NO_OUTPUT_BANDS, 1)
predicted_raster = model.predict({'x': x, 'x1': x1}).reshape(32, 32, NO_OUTPUT_BANDS).T

bands = [0, 1, 2]
predicted_rgb = get_bands_from_array(predicted_raster, bands)
plot_3_band_image(predicted_rgb, title='Predicted Image')

x_rgb = get_bands_from_array(x_raster, bands)
plot_3_band_image(x_rgb, title='Input Image x')
x_rgb = get_bands_from_array(x1_raster, bands)
plot_3_band_image(x_rgb, title='Input Image x1')

y_rgb = get_bands_from_array(y_raster, [15, 29, 47])
plot_3_band_image(y_rgb, title='Original Image')

# tf.keras.utils.plot_model(model, to_file='model_graph.png', show_shapes=True)

for i in range(len(model.layers)):
    print(model.layers[i].name, i)

def plot_detail_branch():
    get_layer_output = (lambda j: K.function(inputs=model.layers[j].input, outputs=model.layers[j].output))
    padded = get_layer_output(1)(x) # TODO: is this the wrong input lol?
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


plot_detail_branch()
