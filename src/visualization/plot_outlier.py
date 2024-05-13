import os
from scipy import stats
from matplotlib import pyplot as plt
import rasterio
import pandas as pd
import numpy as np

from src.visualization.helpers import get_bands_from_raster
from src.visualization.plot_raster import plot_3_band_image, create_rgb_norm


def plot_size_histogram(df, quantile=0.05):
    """plot histogram of file sizes and mark outliers with red color"""
    quantile_size = df['size'].quantile(quantile)
    outlier = df[df['size'] < quantile_size]
    print(outlier)
    plt.xlabel('File size in MB')
    plt.ylabel('Number of files')
    plt.hist(outlier['size'] / 1024 / 1024, bins=60, density=False, alpha=0.75, color='r')
    df = df[df['size'] > quantile_size]
    plt.hist(df['size'] / 1024 / 1024, bins=60, density=False, alpha=0.75, color='b')
    plt.savefig(os.getcwd() + '/../../output/file_size_histogram.png')
    plt.show()


def plot_corresponding_scenes(s_dir, e_dir, output_dir):
    """plot sentinel outliers and corresponding enmap files"""
    s_bands = [2, 3, 4]
    e_bands = [30, 48, 74]
    s_file_list = os.listdir(s_dir)
    e_file_list = os.listdir(e_dir)
    for s_file in s_file_list:
        _figure, axis = plt.subplots(1, 2, figsize=(10, 5))
        timestamp = s_file.split('_')[0]
        e_file = [x for x in e_file_list if timestamp in x][0]
        s_raster = rasterio.open(s_dir + s_file)
        rgb_norm = create_rgb_norm((s_raster.read(s_bands[0]), s_raster.read(s_bands[1]), s_raster.read(s_bands[2])))
        axis[0].imshow(rgb_norm, cmap='viridis')
        axis[0].set_title(s_file)
        e_raster = rasterio.open(e_dir + e_file)
        rgb_norm = create_rgb_norm((e_raster.read(e_bands[0]), e_raster.read(e_bands[1]), e_raster.read(e_bands[2])))
        axis[1].imshow(rgb_norm, cmap='viridis')
        axis[1].set_title(e_file)
        plt.savefig(output_dir + timestamp + '.png')
        plt.show()


# size_df = pd.read_pickle(os.getcwd() + '/../../output/figures/broken_files/Sentinel/file_size_df.pkl')
# size_df = size_df[size_df['file'].str.contains('_spectral.tif')]
# plot_size_histogram(size_df, 0.05)

sentinel_dir = os.getcwd() + '/../../data/preprocessing/Sentinel2_outlier/sentinel/'
enmap_dir = os.getcwd() + '/../../data/preprocessing/Sentinel2_outlier/enmap/'
save_dir = os.getcwd() + '/../../output/figures/broken_files/Sentinel/'

plot_corresponding_scenes(sentinel_dir, enmap_dir, output_dir=save_dir)

# scene_path = enmap_dir + '20240423T071734Z_enmap_spectral.tif'
# # scene_path = enmap_dir + '20240421T013735Z_enmap_spectral.tif'
# raster = rasterio.open(scene_path)
# his_data = raster.read().flatten()
# plt.hist(his_data, bins=100, alpha=0.75, color='b')
# plt.show()