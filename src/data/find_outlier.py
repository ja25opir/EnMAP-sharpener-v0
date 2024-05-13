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

def get_outlier(df, quantile):
    quantile_size = df['size'].quantile(quantile)
    return df[df['size'] < quantile_size]

sentinel_dir = os.getcwd() + '/../../data/preprocessing/Sentinel2/'
# sentinel_dir = '/data/preprocessing/Sentinel2/'
enmap_dir = os.getcwd() + '/../../data/preprocessing/EnMAP/'
outlier_dir = os.getcwd() + '/../../data/preprocessing/Sentinel2_outlier/'

size_df = get_size_df(sentinel_dir)
outlier_df = get_outlier(size_df, 0.05)

for index, row in outlier_df.iterrows():
    shutil.copyfile(sentinel_dir + row['file'], outlier_dir + row['file'])
    timestamp = row['file'].split('_')[0]
    enmap_file = timestamp + '_enmap_spectral.tif'
    shutil.copyfile(enmap_dir + enmap_file, outlier_dir + enmap_file)


