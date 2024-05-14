import os
from scipy import stats
from pandas import DataFrame
import shutil


def get_size_df(path):
    file_list = os.listdir(path)
    size_list = []
    for file in file_list:
        size = os.stat(path + file).st_size
        size_list.append([file, size])

    df = DataFrame({'file': [x[0] for x in size_list], 'size': [x[1] for x in size_list]})
    # only keep spectral files in df
    df = df[df['file'].str.contains('_spectral.tif')]

    # save df
    df.to_pickle(os.getcwd() + '/../../output/file_size_df.pkl')
    return df

def get_outliers(df, quantile):
    quantile_size = df['size'].quantile(quantile)
    return df[df['size'] < quantile_size]

def copy_outliers(target_dir, corresponding_dir, output_dir, corr_suffix='_enmap_spectral.tif'):
    size_df = get_size_df(target_dir)
    outlier_df = get_outliers(size_df, 0.05)
    for index, row in outlier_df.iterrows():
        shutil.copyfile(target_dir + row['file'], output_dir + row['file'])
        timestamp = row['file'].split('_')[0]
        corresponding_file = timestamp + corr_suffix
        shutil.copyfile(corresponding_dir + corresponding_file, output_dir + corresponding_file)



sentinel_dir = os.getcwd() + '/../../data/preprocessing/Sentinel2/'
enmap_dir = os.getcwd() + '/../../data/preprocessing/EnMAP/'

# find sentinel outliers
# outlier_dir = os.getcwd() + '/../../data/preprocessing/Sentinel2_outlier/'
# copy_outliers(sentinel_dir, enmap_dir, outlier_dir)
# find enmap outliers
outlier_dir = os.getcwd() + '/../../data/preprocessing/EnMAP_outlier/'
copy_outliers(enmap_dir, sentinel_dir, outlier_dir, corr_suffix='_sentinel_spectral.tif')


