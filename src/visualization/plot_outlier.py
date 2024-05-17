import os
from scipy import stats
from matplotlib import pyplot as plt
import rasterio
import pandas as pd
import numpy as np

from src.visualization.helpers import get_bands_from_raster
from src.visualization.plot_raster import plot_3_band_image, create_rgb_norm


def plot_size_histogram(df, quant=0.05):
    """plot histogram of file sizes and mark outliers with red color"""
    quantile_size = df['size'].quantile(quant)
    outlier = df[df['size'] < quantile_size]
    print(len(outlier))
    plt.xlabel('File size in MB')
    plt.ylabel('Number of files')
    plt.hist(outlier['size'] / 1024 / 1024, bins=50, density=False, alpha=0.75, color='r')
    lower_df = df[df['size'] > quantile_size]
    lower_df = lower_df[lower_df['size'] < lower_df['size'].quantile(0.99)]
    upper_df = df[df['size'] > df['size'].quantile(0.99)]

    plt.hist(lower_df['size'] / 1024 / 1024, bins=20, density=False, alpha=0.75, color='b')
    plt.hist(upper_df['size'] / 1024 / 1024, bins=12, density=False, alpha=0.75, color='r')
    print(len(upper_df))
    plt.savefig(os.getcwd() + '/../../output/file_size_histogram.png')
    plt.show()


def plot_corresponding_scenes(s_dir, e_dir, output_dir):
    """plot sentinel outliers and corresponding enmap files"""
    s_bands = [2, 3, 4]
    e_bands = [30, 48, 74]
    s_file_list = os.listdir(s_dir)
    e_file_list = os.listdir(e_dir)
    for s_file in s_file_list:
        figure, axis = plt.subplots(2, 2, figsize=(10, 8),
                                    gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
        timestamp = s_file.split('_')[0]
        e_file = [x for x in e_file_list if timestamp in x][0]
        s_raster = rasterio.open(s_dir + s_file)
        e_raster = rasterio.open(e_dir + e_file)
        rgb_norm = create_rgb_norm((s_raster.read(s_bands[0]), s_raster.read(s_bands[1]), s_raster.read(s_bands[2])))

        # plot rgb images
        axis[0][0].imshow(rgb_norm, cmap='viridis')
        axis[0][0].set_title(s_file, pad=20)
        rgb_norm = create_rgb_norm((e_raster.read(e_bands[0]), e_raster.read(e_bands[1]), e_raster.read(e_bands[2])))
        axis[0][1].imshow(rgb_norm, cmap='viridis')
        axis[0][1].set_title(e_file, pad=20)

        # plot histograms
        s_his_data = s_raster.read().flatten()
        s_y_max = np.histogram(s_his_data, bins=100)[0].max()
        e_his_data = e_raster.read().flatten()
        e_y_max = np.histogram(e_his_data, bins=100)[0].max()
        y_max = max(s_y_max, e_y_max)
        axis[1][0].set_ylim(0, y_max)
        axis[1][0].hist(s_his_data, bins=100, alpha=0.75, color='b')
        axis[1][0].tick_params(labelsize=9)
        axis[1][1].set_ylim(0, y_max)
        axis[1][1].hist(e_his_data, bins=100, alpha=0.75, color='b')
        axis[1][1].tick_params(labelsize=9)

        figure.tight_layout()
        plt.savefig(output_dir + timestamp + '.png')
        plt.show()


# size_df = pd.read_pickle(os.getcwd() + '/../../output/figures/broken_files/Sentinel/file_size_df.pkl')
# size_df = pd.read_pickle(os.getcwd() + '/../../output/figures/broken_files/EnMAP/file_size_df.pkl')
# size_df = size_df[size_df['file'].str.contains('_spectral.tif')]
# plot_size_histogram(size_df, 0.05) # Sentinel
# plot_size_histogram(size_df, 0.04) # EnMAP

# Sentinel outlier
# sentinel_dir = os.getcwd() + '/../../data/preprocessing/Sentinel2_outlier/sentinel/'
# enmap_dir = os.getcwd() + '/../../data/preprocessing/Sentinel2_outlier/enmap/'
# save_dir = os.getcwd() + '/../../output/figures/broken_files/Sentinel/'
# plot_corresponding_scenes(sentinel_dir, enmap_dir, output_dir=save_dir)

# EnMAP outlier
sentinel_dir = os.getcwd() + '/../../data/preprocessing/EnMAP_outlier/sentinel/'
enmap_dir = os.getcwd() + '/../../data/preprocessing/EnMAP_outlier/enmap/'
save_dir = os.getcwd() + '/../../output/figures/broken_files/EnMAP/'
plot_corresponding_scenes(sentinel_dir, enmap_dir, output_dir=save_dir)

# todo: plotting looks very blue (green missing)