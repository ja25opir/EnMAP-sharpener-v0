import os, re
from matplotlib import pyplot as plt
import rasterio
import numpy as np


def plot_size_histogram(df, figures_dir, quant=0.05):
    """plot histogram of file sizes and mark outliers with red color"""
    quantile_size = df['size'].quantile(quant)
    outlier = df[df['size'] < quantile_size]
    lower_df = df[df['size'] > quantile_size]
    lower_df = lower_df[lower_df['size'] < lower_df['size'].quantile(0.99)]
    upper_df = df[df['size'] > df['size'].quantile(0.99)]

    bins = np.histogram(
        np.hstack((outlier['size'] / 1024 / 1024, lower_df['size'] / 1024 / 1024, upper_df['size'] / 1024 / 1024)),
        bins=100)[1]
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    plt.xlabel('File size in MB')
    plt.ylabel('Number of files')
    plt.hist(lower_df['size'] / 1024 / 1024, bins=bins, density=False, alpha=0.75, color='b')
    plt.hist(outlier['size'] / 1024 / 1024, bins=bins, density=False, alpha=0.75, color='r')
    # plt.hist(upper_df['size'] / 1024 / 1024, bins=bins, density=False, alpha=0.75, color='r')

    plt.savefig(figures_dir + 'file_size_histogram.png')
    # plt.show()


def plot_corresponding_scenes(input_dir, output_dir, outlier='sentinel', corresponding='enmap'):
    """plot outliers and corresponding files"""
    s_bands = [1, 2, 3]
    e_bands = [16, 30, 48]
    if outlier == 'enmap':
        s_bands = [16, 30, 48]
        e_bands = [1, 2, 3]
    file_list = os.listdir(input_dir)
    for s_file in file_list:
        if not re.search('_' + outlier + '_spectral.tif', s_file):
            continue
        figure, axis = plt.subplots(2, 2, figsize=(10, 8),
                                    gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
        timestamp = s_file.split('_')[0]
        s_raster = rasterio.open(input_dir + s_file)
        e_file = timestamp + '_' + corresponding + '_spectral.tif'
        e_raster = rasterio.open(input_dir + e_file)

        # plot rgb images
        max_reflectance = 1000
        norm_band = (lambda x: np.clip(x, 0, max_reflectance) / max_reflectance * 255)
        rgb = [s_raster.read(s_bands[0]), s_raster.read(s_bands[1]), s_raster.read(s_bands[2])]
        rgb_norm = np.dstack((norm_band(rgb[0]), norm_band(rgb[1]), norm_band(rgb[2]))).astype(np.uint8)
        axis[0][0].imshow(rgb_norm, cmap='viridis', vmin=0, vmax=255)
        axis[0][0].set_title(s_file, pad=20)
        rgb = [e_raster.read(e_bands[0]), e_raster.read(e_bands[1]), e_raster.read(e_bands[2])]
        rgb_norm = np.dstack((norm_band(rgb[0]), norm_band(rgb[1]), norm_band(rgb[2]))).astype(np.uint8)
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
        # plt.show()
