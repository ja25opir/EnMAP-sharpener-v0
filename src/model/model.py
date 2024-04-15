import tensorflow as tf
from tensorflow.keras import layers, models
from matplotlib import pyplot as plt

from src.model.load_data import DataGenerator

tile_size = 100
no_input_bands = 224 + 4
no_output_bands = 224
kernel_size = (9, 9)
kernel_size_internal = (5, 5)
# input_shape = tf.keras.Input(shape=(tile_size, tile_size, input_bands))

# input shape: https://stackoverflow.com/questions/60157742/convolutional-neural-network-cnn-input-shape
# SRCNN: https://github.com/Lornatang/SRCNN-PyTorch/blob/main/model.py 64 - 32 - 1 (no. bands)
# SRCNN: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7115171
# Sentinel CNN: https://github.com/jensleitloff/CNN-Sentinel/blob/master/py/02_train_rgb_finetuning.py

# padding = same (output size = input size) --> rethink this
# activation function relu, relu, linear (Masi) --> rethink this
# layers described in Masi p.4 (2.2) 64 - 32 - 3 (no. bands) kernels: 9x9, 1x1 (3x3), 5x5
model = models.Sequential()
model.add(layers.Conv2D(512, kernel_size,
                        activation='relu',
                        input_shape=(tile_size, tile_size, no_input_bands),
                        padding='same'))
model.add(layers.Conv2D(256, kernel_size_internal, activation='relu', padding='same'))
model.add(layers.Conv2D(no_output_bands, kernel_size_internal, activation='linear', padding='same'))

model.summary()

loss = 'mean_squared_error'  # todo

TRAIN_DATA_DIR = '../../data/preprocessing/model_input/'
BATCH_SIZE = 32  # (Masi: 128)

train_generator = DataGenerator(TRAIN_DATA_DIR,
                                batch_size=BATCH_SIZE,
                                output_size=(tile_size, tile_size),
                                no_input_bands=no_input_bands,
                                no_output_bands=no_output_bands,
                                shuffle=False)

# todo
# validation_generator = DataGenerator(dir_missing,
#                                      batch_size=32,
#                                      output_size=(tile_size, tile_size),
#                                      no_input_bands=no_input_bands,
#                                      no_output_bands=no_output_bands,
#                                      shuffle=False)

model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

history = model.fit(train_generator, epochs=10, verbose=1)
plt.plot(history.history['accuracy'])
plt.show()
