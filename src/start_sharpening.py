import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt

from src.visualization.plot_raster import plot_3_band_image

test_tile = np.load(os.getcwd() + '/../data/preprocessing/model_input/x/20220627T104548Z_0_0.npy')
model_path = os.getcwd() + '/../output/models/first_model.keras'

print(test_tile.shape)
print(test_tile.T.shape)
X_test = np.empty((1, 100, 100, 228))
X_test[0,] = test_tile.T

model = tf.keras.models.load_model(model_path)

print(model.summary())

prediction = model.predict(X_test)

bands = [50, 100, 150]
plot_3_band_image(prediction[0].T, title='Prediction', cmap='viridis')
