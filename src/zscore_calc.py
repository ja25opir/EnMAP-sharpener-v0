import os
from scipy import stats
from pandas import DataFrame
from matplotlib import pyplot as plt

# sentinel_dir = os.getcwd() + '/data/preprocessing/Sentinel2/'
sentinel_dir = '../data/preprocessing/Sentinel2/'
file_list = os.listdir(sentinel_dir)
size_list = []
for file in file_list:
    size = os.stat(sentinel_dir + file).st_size
    size_list.append([file, size])

df = DataFrame({'file': [x[0] for x in size_list], 'size': [x[1] for x in size_list]})

# histo_df = df['size'].describe()
# maybe use quantiles to filter out outliers instead of zscore:
# https://de.wikipedia.org/wiki/Empirisches_Quantil
size_df = df['size'] / 1024 / 1024  # convert to MB
# set axis title
plt.xlabel('File size in MB')
plt.ylabel('Number of files')
plt.hist(size_df, bins=100, density=False, alpha=0.75, color='b')
plt.savefig(os.getcwd() + '/output/file_size_histogram.png')
# plt.show()

# only keep spectral files in df
# df = df[df['file'].str.contains('_spectral.tif')]
#
# # calc zscore and add as new column
# df['zscore'] = stats.zscore(df['size'])
# print(df)
#
# # remove rows with zscore < 0, save as outlier df
# outlier = df[df['zscore'] < -3]
# print(outlier)
# print('number of outliers:', len(outlier))

# plt.hist(zscore, bins=100, density=False, alpha=0.75, color='r')
# plt.show()
