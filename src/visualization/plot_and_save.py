import rasterio
from matplotlib import pyplot as plt
import os, re

from src.visualization.helpers import get_bands_from_raster
from src.visualization.plot_raster import create_rgb_norm


def plot_and_save_enmap_spectral(bands):
    for dir in os.walk('../../data/EnMAP'):
        if re.search("EnMAP/ENMAP01.*", dir[0]):
            for file in dir[2]:
                if re.search(".*SPECTRAL_IMAGE.TIF$", file):
                    raster = rasterio.open(dir[0] + '/' + file)
                    rgb_bands = get_bands_from_raster(raster, bands)
                    rgb_norm = create_rgb_norm((rgb_bands[0], rgb_bands[1], rgb_bands[2]))
                    plt.imshow(rgb_norm, cmap='brg')
                    # plt.show()
                    plt.savefig(OUTPUT_PATH + file.strip('.tif') + '.png', bbox_inches='tight')


def plot_and_save(path, suffix, bands):
    for dir in os.walk(path):
        for file in dir[2]:
            if re.search(".*" + suffix, file):
                raster = rasterio.open(dir[0] + '/' + file)
                rgb_bands = get_bands_from_raster(raster, bands)
                rgb_norm = create_rgb_norm((rgb_bands[0], rgb_bands[1], rgb_bands[2]))
                plt.imshow(rgb_norm, cmap='brg')
                # plt.show()
                plt.savefig(OUTPUT_PATH + file.strip('.tif') + '.png', bbox_inches='tight')


def plot_and_save_clouds(path, suffix):
    for dir in os.walk(path):
        for file in dir[2]:
            if re.search(".*" + suffix, file):
                raster = rasterio.open(dir[0] + '/' + file)
                plt.imshow(raster.read(1), cmap='viridis')
                plt.show()
                # plt.savefig(OUTPUT_PATH + file.strip('.tif') + '.png', bbox_inches='tight')

# Example usage:
ENMAP_DIR_PATH = '../../data/model_input/EnMAP/'
SENTINEL_DIR_PATH = '../../data/model_input/Sentinel2/'
CLOUD_MASKS_PATH = '../../data/model_input/cloud_masks/'
OUTPUT_PATH = '../../output/visualization/showcase/'

# bands_enmap = [16, 30, 48, 72]
bands_enmap = [50, 100, 150]
bands_sentinel = [1, 2, 3]

# plot_and_save_enmap_spectral(bands_enmap)
# plot_and_save(ENMAP_DIR_PATH, 'enmap_spectral.tif', bands_enmap)
# plot_and_save(SENTINEL_DIR_PATH, 'spectral.tif', bands_sentinel)
# plot_and_save_clouds(ENMAP_DIR_PATH, 'enmap_cloud_mask.tif')
# plot_and_save_clouds(ENMAP_DIR_PATH, 'enmap_cloud_mask_upsampled.tif')
# plot_and_save_clouds(SENTINEL_DIR_PATH, 'cloud_mask.tif')
plot_and_save_clouds(CLOUD_MASKS_PATH, 'cloud_mask_combined.tif')