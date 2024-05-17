import os, random
import rasterio
import tensorflow as tf
import numpy as np

from src.visualization.helpers import get_bands_from_array
from src.visualization.plot_raster import plot_3_band_image

x_data_path = os.getcwd() + '/../../data/preprocessing/model_input/x/'
y_data_path = os.getcwd() + '/../../data/preprocessing/model_input/y/'
model_path = os.getcwd() + '/../../output/models/'

x_data = os.listdir(x_data_path)
y_data = os.listdir(y_data_path)

# get random file from data
random_file = random.choice(y_data)

x_raster = np.load(x_data_path + random_file)
y_raster = np.load(y_data_path + random_file)

model = tf.keras.models.load_model(model_path + 'first_model.keras')

print(model.summary())

predicted_raster = model.predict(x_raster.T.reshape(1, 100, 100, 228)).reshape(100, 100, 224).T

bands = [50, 100, 150]
predicted_rgb = get_bands_from_array(predicted_raster, bands)
plot_3_band_image(predicted_rgb, title='Predicted Image')

x_rgb = get_bands_from_array(x_raster, bands)
plot_3_band_image(x_rgb, title='Input Image')

y_rgb = get_bands_from_array(y_raster, bands)
plot_3_band_image(y_rgb, title='Original Image')

# todo: for a real prediction we have to upscale an enmap image that was not used for training, and add sentinel bands
