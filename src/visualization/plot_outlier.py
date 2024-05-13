import os
from scipy import stats
from matplotlib import pyplot as plt
import rasterio
import pandas as pd
import numpy as np

from src.visualization.helpers import get_bands_from_raster
from src.visualization.plot_raster import plot_3_band_image, create_rgb_norm

size_df = pd.read_pickle(os.getcwd() + '/../../output/figures/broken_files/file_size_df.pkl')
size_df = size_df[size_df['file'].str.contains('_spectral.tif')]

plt.xlabel('File size in MB')
plt.ylabel('Number of files')
# print(size_df)

# todo: draw histogram and save
# todo: filter out outliers with percentile (zscore before)
# todo: plot comparison of outliers again

quantile = size_df['size'].quantile(0.05)
outlier = size_df[size_df['size'] < quantile]
print(outlier)
plt.hist(outlier['size'] / 1024 / 1024, bins=60, density=False, alpha=0.75, color='r')
size_df = size_df[size_df['size'] > quantile]
plt.hist(size_df['size'] / 1024 / 1024, bins=60, density=False, alpha=0.75, color='b')
plt.show()
plt.savefig(os.getcwd() + '/../../output/file_size_histogram.png')

"""plot sentinel outliers and corresponding enmap files"""
# sentinel_dir = os.getcwd() + '/../test/'
# enmap_dir = os.getcwd() + '/../teste/'
# s_file_list = os.listdir(sentinel_dir)
# e_file_list = os.listdir(enmap_dir)
#
# s_bands = [2, 3, 4]
# e_bands = [30, 48, 74]
# for s_file in s_file_list:
#     _figure, axis = plt.subplots(1, 2, figsize=(10, 5))
#     timestamp = s_file.split('_')[0]
#     e_file = [x for x in e_file_list if timestamp in x][0]
#     s_raster = rasterio.open(sentinel_dir + s_file)
#     rgb_norm = create_rgb_norm((s_raster.read(s_bands[0]), s_raster.read(s_bands[1]), s_raster.read(s_bands[2])))
#     axis[0].imshow(rgb_norm, cmap='viridis')
#     axis[0].set_title(s_file)
#     e_raster = rasterio.open(enmap_dir + e_file)
#     rgb_norm = create_rgb_norm((e_raster.read(e_bands[0]), e_raster.read(e_bands[1]), e_raster.read(e_bands[2])))
#     axis[1].imshow(rgb_norm, cmap='viridis')
#     axis[1].set_title(e_file)
#     plt.savefig(f'{timestamp}.png')
#     # plt.show()
#
#     # raster = rasterio.open(sentinel_dir + file)
#     # rgb_bands = get_bands_from_raster(raster, s_bands)
#     # plot_3_band_image(rgb_bands, file, cmap='viridis')
