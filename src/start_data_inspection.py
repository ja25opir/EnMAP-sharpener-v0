import os
import pandas as pd

from data.find_outlier import copy_outliers
from visualization.plot_outlier import plot_size_histogram, plot_corresponding_scenes

SENTINEL_DIR = os.getcwd() + '/data/preprocessing/Sentinel2/'
ENMAP_DIR = os.getcwd() + '/data/preprocessing/EnMAP/'

if __name__ == '__main__':
    # find Sentinel outliers
    sentinel_outlier_dir = os.getcwd() + '/data/preprocessing/Sentinel2_outlier/'
    sentinel_figures_dir = os.getcwd() + '/output/figures/broken_files/Sentinel/'
    copy_outliers(SENTINEL_DIR, ENMAP_DIR, sentinel_outlier_dir, sentinel_figures_dir, quantile=0.07)

    # find EnMAP outliers
    enmap_outlier_dir = os.getcwd() + '/data/preprocessing/EnMAP_outlier/'
    enmap_figures_dir = os.getcwd() + '/output/figures/broken_files/EnMAP/'
    copy_outliers(ENMAP_DIR, SENTINEL_DIR, enmap_outlier_dir, enmap_figures_dir, corr_suffix='_sentinel_spectral.tif',
                  quantile=0.1)

    # plot Sentinel outlier
    sentinel_size_df = pd.read_pickle(sentinel_figures_dir + '/file_size_df.pkl')
    sentinel_size_df = sentinel_size_df[sentinel_size_df['file'].str.contains('_spectral.tif')]
    # plot_size_histogram(sentinel_size_df, sentinel_figures_dir, quant=0.07)
    # plot_corresponding_scenes(sentinel_outlier_dir, output_dir=sentinel_figures_dir)

    # plot EnMAP outlier
    enmap_size_df = pd.read_pickle(enmap_figures_dir + 'file_size_df.pkl')
    enmap_size_df = enmap_size_df[enmap_size_df['file'].str.contains('_spectral.tif')]
    # plot_size_histogram(enmap_size_df, enmap_figures_dir, quant=0.1)
    # plot_corresponding_scenes(enmap_outlier_dir, output_dir=enmap_figures_dir, outlier='enmap', corresponding='sentinel')
