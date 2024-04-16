import rasterio
from src.visualization.helpers import print_raster_info, get_bands_from_raster
from src.visualization.plot_raster import plot_3_band_image

# DIR_PATH = "../../data/Sentinel2/S2A_MSIL2A_20230811T101031_N0509_R022_T33UUS_20230811T161756.SAFE/GRANULE/L2A_T33UUS_A042490_20230811T101028/IMG_DATA/R10m/"
# FILE_NAME = "T33UUS_20230811T101031"

# todo: stack bands (given files (despite TCI) only contain one band)
# origin = rasterio.open(DIR_PATH + FILE_NAME + '_AOT_10m.jp2')
# origin = rasterio.open(DIR_PATH + FILE_NAME + '_B02_10m.jp2')
# origin = rasterio.open(DIR_PATH + FILE_NAME + '_B03_10m.jp2')
# origin = rasterio.open(DIR_PATH + FILE_NAME + '_B04_10m.jp2')
# origin = rasterio.open(DIR_PATH + FILE_NAME + '_B08_10m.jp2')
# origin = rasterio.open(DIR_PATH + FILE_NAME + '_TCI_10m.jp2') # true color image (B02 + B03 + B04)
# origin = rasterio.open(DIR_PATH + FILE_NAME + '_WVP_10m.jp2')

# DIR_PATH = "../../data/Sentinel2/scraped/"
# FILE_NAME = "Sentinel_ENMAP01-____L2A-DT0000001280_20220627T104548Z_012_V010400_20231124T152718Z-SPECTRAL_IMAGE.TIF.tiff"
DIR_PATH = "../../data/Sentinel2/scraped/"
# FILE_NAME = 'Sentinel_EnMAP_cropped_spectral.tif.tiff'
IGNORE_PATH = "../../data/Sentinel2/scraped/tests"
IGNORE_PATH2 = "../../data/Sentinel2/scraped/zulip"

# for directory in os.walk(DIR_PATH):
#     if directory[0] != IGNORE_PATH and directory[0] != IGNORE_PATH2:
#         for filename in directory[2]:
#             print('filename', filename)
#             origin = rasterio.open(DIR_PATH + filename)
#             print_raster_info(origin)
#             selected_bands = [1, 2, 3]
#             band_array = get_bands_from_raster(origin, selected_bands)
#             plot_3_band_image(band_array)

FILE_NAME = "Sentinel_dn_int16.tiff"
# FILE_NAME = "Sentinel_EnMAP_cropped_spectral.tif.tiff"
# origin = rasterio.open(DIR_PATH + FILE_NAME)
origin = rasterio.open(
    "../../data/EnMAP/ENMAP01-____L2A-DT0000001280_20220627T104548Z_012_V010400_20231124T152718Z/EnMAP_cropped_spectral.tif")
print_raster_info(origin)
# selected_bands = [1, 2, 3, 4] # Sentinel bands
selected_bands = [16, 30, 48, 72] # EnMAP bands matching Sentinel bands
band_array = get_bands_from_raster(origin, selected_bands)

# prints values of selected bands at a given coordinate
sample = list(origin.sample([(315000, 5687000)]))[0]
print('490nm:', sample[selected_bands[0] - 1])
print('560nm:', sample[selected_bands[1] - 1])
print('665nm:', sample[selected_bands[2] - 1])
print('842nm:', sample[selected_bands[3] - 1])

plot_3_band_image(band_array)
