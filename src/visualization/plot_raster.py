import numpy as np
from matplotlib import pyplot as plt


def create_rgb_norm(rgb_bands):
    max_val = np.amax([np.amax(rgb_bands[0]), np.amax(rgb_bands[1]), np.amax(rgb_bands[2])])
    # normalize all values into 0...255 values and stack three bands into a 3d-array (= rgb-image)
    return (
        np.dstack((rgb_bands[0] / max_val * 256, rgb_bands[1] / max_val * 256, rgb_bands[2] / max_val * 256))).astype(
        np.uint8)


def plot_3_band_image(bands):
    rgb_norm = create_rgb_norm((bands[0], bands[1], bands[2]))
    plt.imshow(rgb_norm, cmap='viridis')
    plt.show()

