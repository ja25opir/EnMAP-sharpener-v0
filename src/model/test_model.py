import os, random
import rasterio
import tensorflow as tf
import numpy as np
import keras.backend as K

from src.model.architecture import ReflectionPadding2D, ReflectionPadding3D, SFTLayer
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
                  'SFTLayer': SFTLayer}
model = tf.keras.models.load_model(model_path + 'masi_3_3.keras', custom_objects=custom_objects)

print(model.summary())

x_raster = x_raster[(50, 100, 150, 225, 226, 227), :, :] # 6 bands only
model_input = x_raster.T.reshape(1, 32, 32, 6)
predicted_raster = model.predict(model_input).reshape(32, 32, 3).T
# x = x_raster.T.reshape(1, 32, 32, 228)
# x1 = x1_raster.T.reshape(1, 32, 32, 224)
# predicted_raster = model.predict([x, x1]).reshape(32, 32, 224).T
# predicted_raster = model.predict(x1).reshape(32, 32, 224).T

# bands = [50, 100, 150]
bands = [0,1,2]
predicted_rgb = get_bands_from_array(predicted_raster, bands)
plot_3_band_image(predicted_rgb, title='Predicted Image')

x_rgb = get_bands_from_array(x_raster, bands)
plot_3_band_image(x_rgb, title='Input Image')

y_rgb = get_bands_from_array(y_raster, [50,100,150])
plot_3_band_image(y_rgb, title='Original Image')

# tf.keras.utils.plot_model(model, to_file='model_graph.png', show_shapes=True)

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
