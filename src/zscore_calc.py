import os
from scipy import stats
from pandas import DataFrame

sentinel_dir = os.getcwd() + '/data/preprocessing/Sentinel2/'
file_list = os.listdir(sentinel_dir)
size_list = []
for file in file_list:
    size = os.stat(sentinel_dir + file).st_size
    size_list.append([file, size])

df = DataFrame({'file': [x[0] for x in size_list], 'size': [x[1] for x in size_list]})

# remove cloud masks from df
df = df[df['file'].str.contains('_cloud_mask.tif')]

# calc zscore and add as new column
df['zscore'] = stats.zscore(df['size'])

# remove rows with zscore < 0, save as outlier df
outlier = df[df['zscore'] < 0]
print(outlier)
print('number of outliers:', len(outlier))

# spectral_list = []
# for file in file_list:
#     if '_spectral.tif' in file:
#         spectral_list.append(file)
#

#
# zscore = stats.zscore(size_list)
# standard_deviation = stats.tstd(size_list)
# print(size_list)
# for i in range(len(zscore)):
#     if zscore[i] < 0:
#         print(file_list[i], size_list[i])
# plt.hist(size_list, bins=100, density=False, alpha=0.75, color='b')
# plt.show()
# plt.hist(zscore, bins=100, density=False, alpha=0.75, color='r')
# plt.show()
