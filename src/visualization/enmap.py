import os
import re
import rasterio
from matplotlib import pyplot as plt
from rasterio.plot import show
import rasterio.mask

from src.data.helpers import get_bands_from_array, get_bands_from_raster, print_raster_info
from src.data.preprocess import crop_raster, get_bounding_box_from_xml, get_inscribed_rect_from_bbox
from src.visualization.plot_raster import create_rgb_norm, plot_3_band_image


def use_pyplot(save=False):
    """
    Visualize all EnMAP directories inside a root directory one plot with subplots per EnMAP-dir using matplotlib.pyplot.
    :param save: toggle saving the plot
    :return:
    """
    for dir in os.walk('../../data/EnMAP'):
        if re.search("EnMAP/ENMAP01.*", dir[0]):
            print("dir:", dir[0])
            _figure, axis = plt.subplots(3, 4, figsize=(24, 18))
            i, j = 0, 0
            for file in dir[2]:
                if re.search(".*.TIF$", file):
                    filename = file.split('Z-')[1]
                    print(filename)
                    img = rasterio.open(dir[0] + '/' + file)
                    axis[i, j].imshow(img.read(1), cmap='viridis')  # reads first band only
                    axis[i, j].set_title(filename)
                    if i < 2:
                        i = i + 1
                    else:
                        i = 0
                        j = j + 1
            if save:
                plt.savefig('../../output/visualization/EnMAP/' + dir[0] + '.png', bbox_inches='tight')
            plt.show()


def use_rasterio(save=False):
    """
    Visualize all spectral EnMAP images from directories inside a root directory using rasterio.
    :param save: toggle saving the plot
    :return:
    """
    for dir in os.walk('../../data/EnMAP'):
        if re.search("EnMAP/ENMAP01.*", dir[0]):
            print("dir:", dir[0])
            for file in dir[2]:
                if re.search(".*SPECTRAL_IMAGE.TIF$", file):
                    _fig, axis = plt.subplots(1, 1, figsize=(6, 5))
                    filename = file.split('Z-')[1]
                    print(filename)
                    img = rasterio.open(dir[0] + '/' + file)
                    large_raster = img.read(1, masked=True)  # reads first band only
                    show(large_raster, cmap='viridis', title=filename, ax=axis)
                    if save:
                        plt.savefig('../output/visualization/EnMAP/' + dir[0] + '_rasterio.png', bbox_inches='tight')
                    plt.show()


def plot_bands(path, bands, save=False, savename=''):
    """
    Plot three given bands as combined RGB image.
    :param path: path to EnMAP file (.tif)
    :param bands: array containing three band numbers
    :param save: toggle saving the plot
    :param savename: name of the saved file
    :return:
    """
    img = rasterio.open(path)
    r_band = img.read(bands[0])
    g_band = img.read(bands[1])
    b_band = img.read(bands[2])
    _figure, axis = plt.subplots(1, 4, figsize=(24, 18))
    axis[0].imshow(r_band, cmap='viridis')
    axis[0].set_title('band no. ' + str(bands[0]))
    axis[1].imshow(g_band, cmap='viridis')
    axis[1].set_title('band no. ' + str(bands[1]))
    axis[2].imshow(b_band, cmap='viridis')
    axis[2].set_title('band no. ' + str(bands[2]))
    rgb_norm = create_rgb_norm((r_band, g_band, b_band))
    axis[3].imshow(rgb_norm, cmap='viridis')
    axis[3].set_title('combined')
    if save:
        print("save...")
        plt.savefig('../../output/visualization/EnMAP' + str(savename) + '_' + str(bands) + '.png', bbox_inches='tight')
    plt.show()


# useful wiki: https://automating-gis-processes.github.io/CSC18/lessons/L6/reading-raster.html
# transform coordinates: https://epsg.io/transform#s_srs=32633&t_srs=32633&x=303645.0000000&y=5696295.0000000

# use_rasterio(save=False)
# use_pyplot(save=False)

IMG_PATH = '../../data/EnMAP/ENMAP01-____L2A-DT0000001280_20220627T104548Z_012_V010400_20231124T152718Z/ENMAP01-____L2A-DT0000001280_20220627T104548Z_012_V010400_20231124T152718Z-SPECTRAL_IMAGE.TIF'
XML_PATH = '../../data/EnMAP/ENMAP01-____L2A-DT0000001280_20220627T104548Z_012_V010400_20231124T152718Z/ENMAP01-____L2A-DT0000001280_20220627T104548Z_012_V010400_20231124T152718Z-METADATA.XML'
# IMG_PATH = '../../data/EnMAP/Example_Tyrol/dims_op_oc_oc-en_700949147_4/ENMAP.HSI.L2A/ENMAP-HSI-L2ADT0000001049_28-2022-06-12T10_57_35.160_schwipe-charter_700949145_722423345_2023-09-22T19_27_20/ENMAP01-____L2A-DT0000001049_20220612T105735Z_028_V010303_20230922T131826Z/ENMAP01-____L2A-DT0000001049_20220612T105735Z_028_V010303_20230922T131826Z-SPECTRAL_IMAGE.TIF'
selected_bands = [50, 100, 150]
# plot_bands(IMG_PATH, selected_bands, save=False, savename='ENMAP01-____L2A-DT0000001280_20220627T104548Z_012_V010400_20231124T152718Z-SPECTRAL_IMAGE')

origin = rasterio.open(IMG_PATH)
print_raster_info(origin)
print(origin.profile)
# band_array = get_bands_from_raster(origin, selected_bands) # returns 3 stacked bands
# plot_3_band_image(band_array)

bbox = get_bounding_box_from_xml(XML_PATH)
ir_bbox = get_inscribed_rect_from_bbox(bbox)
print(ir_bbox)

crop_shape = [{'type': 'Polygon',
               'coordinates': [ir_bbox]}]  # GeoJSON format

OUTPUT_DIR = '../../data/EnMAP/ENMAP01-____L2A-DT0000001280_20220627T104548Z_012_V010400_20231124T152718Z/'
# cropped = crop_raster(origin, crop_shape, save=True, output_dir=OUTPUT_DIR, savename='EnMAP_cropped_spectral')
timestamp = re.search('\d{4}\d{2}\d{2}T\d{6}Z', IMG_PATH)
savename = timestamp.group() + '_enmap_cropped_spectral'
cropped = crop_raster(origin, crop_shape, save=False, output_dir=OUTPUT_DIR, savename=savename)
band_array_cropped = get_bands_from_array(cropped[0], selected_bands)
plot_3_band_image(band_array_cropped)

new_origin = rasterio.open(OUTPUT_DIR + savename + '.tif')
print(new_origin.profile)
