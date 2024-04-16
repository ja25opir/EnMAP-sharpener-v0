import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from src.visualization.helpers import get_bands_from_raster, get_bands_from_array


def create_rgb_norm(rgb_bands):
    max_val = np.amax([np.amax(rgb_bands[0]), np.amax(rgb_bands[1]), np.amax(rgb_bands[2])])
    # normalize all values into 0...255 values and stack three bands into a 3d-array (= rgb-image)
    return (
        np.dstack((rgb_bands[0] / max_val * 255, rgb_bands[1] / max_val * 255, rgb_bands[2] / max_val * 255))).astype(
        np.uint8)


def create_zscore_rgb_norm(rgb_bands):
    # zscore over all bands then normalize to 0...255
    zscore = stats.zscore(rgb_bands, axis=None)
    max_val = np.amax([np.amax(zscore[0]), np.amax(zscore[1]), np.amax(zscore[2])])
    return np.dstack(
        (zscore[0] / max_val * 255, zscore[1] / max_val * 255, zscore[2] / max_val * 255)).astype(np.uint8)


def create_band_zscore_rgb_norm(rgb_bands):
    # zscore for each band then normalize to 0...255
    zscore = [stats.zscore(rgb_bands[0], axis=None),
              stats.zscore(rgb_bands[1], axis=None),
              stats.zscore(rgb_bands[2], axis=None)]
    max_val = [np.amax(zscore[0]), np.amax(zscore[1]), np.amax(zscore[2])]
    return np.dstack(
        (zscore[0] / max_val[0] * 255, zscore[1] / max_val[1] * 255, zscore[2] / max_val[2] * 255)).astype(np.uint8)


def plot_3_band_zscore_image(raster, bands, title, cmap='viridis', is_array=False):
    rgb_bands = get_bands_from_array(raster, bands) if is_array else get_bands_from_raster(raster, bands)
    rgb_norm = create_band_zscore_rgb_norm((rgb_bands[0], rgb_bands[1], rgb_bands[2]))
    plt.title(title)
    plt.imshow(rgb_norm, cmap=cmap)
    plt.show()


def plot_3_band_image(bands):
    rgb_norm = create_rgb_norm((bands[0], bands[1], bands[2]))
    plt.imshow(rgb_norm, cmap='viridis')
    plt.show()
