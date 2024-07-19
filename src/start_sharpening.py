import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt

from src.visualization.helpers import get_bands_from_array
from src.visualization.plot_raster import plot_3_band_image

test_tile = np.load(os.getcwd() + '/../data/preprocessing/model_input/x/20220916T104547Z_3_6.npy')
model_path = os.getcwd() + '/../output/models/first_model.keras'

print(test_tile.shape)
print(test_tile.T.shape)
X_test = np.empty((1, 100, 100, 228))
X_test[0,] = test_tile.T

model = tf.keras.models.load_model(model_path)

print(model.summary())

prediction = model.predict(X_test)

predicted_img = prediction[0].T

bands = [50, 100, 150]
rgb_bands = get_bands_from_array(test_tile, bands)
plot_3_band_image(rgb_bands, title='Input', cmap='viridis')
rgb_bands = get_bands_from_array(predicted_img, bands)
plot_3_band_image(predicted_img, title='Prediction', cmap='viridis')
# some structures are already learned!
# but: border effects are huge! -> need to add padding to input data (?)
# clouds have negative values. maybe use mask with negative value also to mask them in real world images?

# train set: use only tiles with < x % cloud cover