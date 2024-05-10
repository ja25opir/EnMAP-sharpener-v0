import os
from scipy import stats

sentinel_dir = os.getcwd() + '/data/preprocessing/Sentinel2/'
file_list = os.listdir(sentinel_dir)

size_list = []
for file in file_list:
    size_list.append(os.stat(sentinel_dir + file).st_size)

zscore = stats.zscore(size_list)
standard_deviation = stats.tstd(size_list)
for i in range(len(zscore)):
    if zscore[i] < 0:
        print(file_list[i], size_list[i])
# plt.hist(size_list, bins=100, density=False, alpha=0.75, color='b')
# plt.show()
# plt.hist(zscore, bins=100, density=False, alpha=0.75, color='r')
# plt.show()
