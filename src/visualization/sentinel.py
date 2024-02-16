import rasterio
from matplotlib import pyplot as plt
from src.data.helpers import print_raster_info, get_bands_from_raster
from src.visualization.plot_raster import plot_3_band_image

DIR_PATH = "../../data/Sentinel2/S2A_MSIL2A_20230811T101031_N0509_R022_T33UUS_20230811T161756.SAFE/GRANULE/L2A_T33UUS_A042490_20230811T101028/IMG_DATA/R10m/"
FILE_NAME = "T33UUS_20230811T101031"

# origin = rasterio.open(DIR_PATH + FILE_NAME + '_AOT_10m.jp2')
# origin = rasterio.open(DIR_PATH + FILE_NAME + '_B02_10m.jp2')
# origin = rasterio.open(DIR_PATH + FILE_NAME + '_B03_10m.jp2')
# origin = rasterio.open(DIR_PATH + FILE_NAME + '_B04_10m.jp2')
# origin = rasterio.open(DIR_PATH + FILE_NAME + '_B08_10m.jp2')
origin = rasterio.open(DIR_PATH + FILE_NAME + '_TCI_10m.jp2') # true color image (B02 + B03 + B04)
# origin = rasterio.open(DIR_PATH + FILE_NAME + '_WVP_10m.jp2')

print_raster_info(origin)

# todo: stack bands? (given files (despite TCI) only contain one band)
selected_bands = [1, 2, 3]
band_array = get_bands_from_raster(origin, selected_bands)
plot_3_band_image(band_array)

# plt.imshow(origin.read(1), cmap='viridis')
# plt.show()
