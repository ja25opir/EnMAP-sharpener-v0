import rasterio
from matplotlib import pyplot as plt
import os, re
import numpy as np

from src.visualization.helpers import get_bands_from_raster
from src.visualization.plot_raster import create_rgb_norm


def plot_and_save_enmap_spectral(bands, title):
    for dir in os.walk('../../data/EnMAP'):
        if re.search("EnMAP/ENMAP01.*", dir[0]):
            for file in dir[2]:
                if re.search(".*SPECTRAL_IMAGE.TIF$", file):
                    print('saving', file)
                    raster = rasterio.open(dir[0] + '/' + file)
                    rgb_bands = get_bands_from_raster(raster, bands)
                    rgb_norm = create_rgb_norm((rgb_bands[0], rgb_bands[1], rgb_bands[2]))
                    plt.imshow(rgb_norm, cmap='brg')
                    plt.title(title)
                    # plt.show()
                    plt.savefig(OUTPUT_PATH + file.strip('.tif') + '.png', bbox_inches='tight')


def plot_and_save(path, suffix, bands, title, brightness_factor=1):
    for dir in os.walk(path):
        for file in dir[2]:
            if re.search(".*" + suffix, file):
                print('saving', file)
                raster = rasterio.open(dir[0] + '/' + file)
                rgb_bands = get_bands_from_raster(raster, bands)
                rgb_norm = create_rgb_norm((rgb_bands[0], rgb_bands[1], rgb_bands[2])) * brightness_factor
                plt.imshow(rgb_norm, cmap='brg')
                plt.title(title)
                # plt.show()
                plt.savefig(OUTPUT_PATH + file.strip('.tif') + '.png', bbox_inches='tight')


def plot_and_save_clouds(path, suffix, title):
    for dir in os.walk(path):
        for file in dir[2]:
            if re.search(".*" + suffix, file):
                print('saving', file)
                raster = rasterio.open(dir[0] + '/' + file)
                plt.imshow(raster.read(1), cmap='viridis')
                plt.title(title)
                # plt.show()
                plt.savefig(OUTPUT_PATH + file.strip('.tif') + '.png', bbox_inches='tight')


# Example usage:
ENMAP_DIR_PATH = '../../data/preprocessing/EnMAP/'
SENTINEL_DIR_PATH = '../../data/preprocessing/Sentinel2/'
CLOUD_MASKS_PATH = '../../data/preprocessing/cloud_masks/'
MASKED_SCENES_PATH = '../../data/preprocessing/masked_scenes/'
OUTPUT_PATH = '../../output/visualization/showcase/'

# bands_enmap = [16, 30, 48, 72]
bands_enmap = [50, 100, 150]
bands_sentinel = [1, 2, 3]

# plot_and_save_enmap_spectral(bands_enmap, 'EnMAP scene (false color)')
# plot_and_save(ENMAP_DIR_PATH, 'enmap_spectral.tif', bands_enmap, 'cropped EnMAP scene')
# plot_and_save(SENTINEL_DIR_PATH, 'spectral.tif', bands_sentinel, 'Sentinel scene from API', brightness_factor=2)
# plot_and_save_clouds(ENMAP_DIR_PATH, 'enmap_cloud_mask.tif', 'EnMAP cloud mask')
# plot_and_save_clouds(ENMAP_DIR_PATH, 'enmap_cloud_mask_upsampled.tif', 'EnMAP cloud mask upsampled to 10x10m/pixel')
# plot_and_save_clouds(SENTINEL_DIR_PATH, 'cloud_mask.tif', 'Sentinel cloud mask')
# plot_and_save_clouds(CLOUD_MASKS_PATH, 'cloud_mask_combined.tif', 'combined cloud mask')
plot_and_save(MASKED_SCENES_PATH, 'enmap_masked.tif', bands_enmap, 'masked EnMAP scene')
plot_and_save(MASKED_SCENES_PATH, 'sentinel_masked.tif', bands_sentinel, 'masked Sentinel scene', brightness_factor=2)
