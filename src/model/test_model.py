import os, random
import rasterio
import tensorflow as tf
import numpy as np
import keras.backend as K

from src.model.architecture import ReflectionPadding2D, ReflectionPadding3D, SFTLayer, DILayer
from src.visualization.helpers import get_bands_from_array
from src.visualization.plot_raster import plot_3_band_image

x_data_path = os.getcwd() + '/../../data/preprocessing/model_input/x/'
x1_data_path = os.getcwd() + '/../../data/preprocessing/model_input/x1/'
y_data_path = os.getcwd() + '/../../data/preprocessing/model_input/y/'
model_path = os.getcwd() + '/../../output/models/'

x_data = os.listdir(x_data_path)
y_data = os.listdir(y_data_path)

# get random file from data
# random_file = random.choice(y_data)
random_file = '20220627T104548Z_0_0.npy'

x_raster = np.load(x_data_path + random_file)
x1_raster = np.load(x1_data_path + random_file)
y_raster = np.load(y_data_path + random_file)

# model = tf.keras.models.load_model(model_path + 'first_model.keras')
# all hidden layers in a deep model only filter cloud + cloud edges
# --> not true: todo: inspect all bands of the hidden layers
custom_objects = {'ReflectionPadding2D': ReflectionPadding2D,
                  'ReflectionPadding3D': ReflectionPadding3D,
                  'SFTLayer': SFTLayer,
                  'DILayer': DILayer}
model = tf.keras.models.load_model(model_path + 'MMSRes.keras', custom_objects=custom_objects)
# model = tf.keras.models.load_model(model_path + 'TestSaPNN_3_3.keras', custom_objects=custom_objects)

print(model.summary())

x_raster = x_raster[(224, 225, 226, 227), :, :]
# x_raster = x_raster[(50, 100, 150), :, :] # 3 bands only
# x1_raster = x1_raster[(15, 29, 47), :, :]  # 3 bands only
x1_raster = x1_raster[(15, 29, 47, 71), :, :]  # 3 bands only
# x1_raster = x1_raster[80:100, :, :]  # 20 bands
# model_input = x_raster.T.reshape(1, 32, 32, 6)
# predicted_raster = model.predict(model_input).reshape(32, 32, 3).T
x = x_raster.T.reshape(1, 32, 32, 4)
x1 = x1_raster.T.reshape(1, 32, 32, 4, 1)
predicted_raster = model.predict([x, x1]).reshape(32, 32, 4).T
# predicted_raster = model.predict(x1).reshape(32, 32, 3).T
# predicted_raster = model.predict(x).reshape(32, 32, 3).T
# predicted_raster = model.predict(x1).reshape(32, 32, 224).T

# bands = [50, 100, 150]
bands = [0, 1, 2]
predicted_rgb = get_bands_from_array(predicted_raster, bands)
plot_3_band_image(predicted_rgb, title='Predicted Image')

x_rgb = get_bands_from_array(x_raster, bands)
plot_3_band_image(x_rgb, title='Input Image x')
x_rgb = get_bands_from_array(x1_raster, bands)
plot_3_band_image(x_rgb, title='Input Image x1')  # todo x1 und x zueinander verschoben!

y_rgb = get_bands_from_array(y_raster, [15, 29, 47])
plot_3_band_image(y_rgb, title='Original Image')

# tf.keras.utils.plot_model(model, to_file='model_graph.png', show_shapes=True)

for i in range(len(model.layers)):
    print(model.layers[i].name, i)

get_layer_output = (lambda j: K.function(inputs=model.layers[j].input, outputs=model.layers[j].output))
padded = get_layer_output(1)(x)
first_2d = get_layer_output(2)(padded)
first_2d_readable = first_2d[0, :, :, :].T
arr = get_bands_from_array(padded[0, :, :, :].T, [0, 1, 2])
plot_3_band_image(arr, title='First 2d pad')
padded = get_layer_output(5)(first_2d)
second_2d = get_layer_output(6)(padded)
padded = get_layer_output(12)(second_2d)
third_2d = get_layer_output(14)(padded)

arr = get_bands_from_array(first_2d[0, :, :, :].T, [0, 1,
                                                    2])  # todo: only extract 3 feat maps with 2d convs and inject (merged = 3 * 64 feature maps for first layer)
plot_3_band_image(arr, title='First 2d conv')
arr = get_bands_from_array(second_2d[0, :, :, :].T, [0, 1, 2])
plot_3_band_image(arr, title='Second 2d conv')
arr = get_bands_from_array(third_2d[0, :, :, :].T, [0, 1, 2])
plot_3_band_image(arr, title='Third 2d conv')

# input_2d_pad = get_layer_output(3)(x)
# arr = get_bands_from_array(input_2d_pad[0, :, :, :].T, bands)
# plot_3_band_image(arr, title='Layer 2d pad')
# output_2d_conv = get_layer_output(5)(input_2d_pad)
# arr = get_bands_from_array(output_2d_conv[0, :, :, :].T, bands)
# plot_3_band_image(arr, title='Layer 2d conv')
#
# input_3d_pad = get_layer_output(2)([x1])
# arr = get_bands_from_array(input_3d_pad[0, :, :, :, 0].T, bands)
# plot_3_band_image(arr, title='Layer 3d pad')
# output_3d_conv = get_layer_output(4)([input_3d_pad])
# arr = get_bands_from_array(output_3d_conv[0, :, :, :, 0].T, bands)
# plot_3_band_image(arr, title='Layer 3d conv')
# todo: layer outputs are empty but predicted image has values

# output_sft = get_layer_output(6)([output_3d_conv, output_2d_conv])
# for i in range(output_sft.shape[-1]):
#     arr = get_bands_from_array(output_sft[0, :, :, :, i].T, bands)
#     plot_3_band_image(arr, title='Layer SFT conv %d' %i)

# output_3d_final = get_layer_output(7)(output_3d_conv)
# arr = get_bands_from_array(output_3d_final[0, :, :, :, 0].T, bands)
# plot_3_band_image(arr, title='Layer 3d final')

# get_layer_output = K.function(inputs=model.layers[0].input, outputs=model.layers[0].output)
# input_l = get_layer_output([x1])
# for i in range(len(model.layers)):
#     if i == 0:
#         get_layer_output = K.function(inputs=model.layers[0].input, outputs=model.layers[0].output)
#         input_l = get_layer_output([x1])
#     else:
#         get_layer_output = K.function(inputs=model.layers[i].input, outputs=model.layers[i].output)
#         output_l = get_layer_output([input_l])
#         input_l = output_l
#         if model.layers[i].name == 'conv3d':
#             arr = get_bands_from_array(output_l[0, :, :, :, 0].T, bands)
#             plot_3_band_image(arr, title='Layer ' + str(i + 1))
