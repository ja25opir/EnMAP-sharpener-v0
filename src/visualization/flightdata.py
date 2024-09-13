import spectral
import matplotlib.pyplot as plt
import rasterio

from src.visualization.helpers import get_bands_from_raster
from src.visualization.plot_raster import create_rgb_norm, plot_3_band_image, plot_3_band_image_clipped


def plot_bands(hdr_path, bands, save=False, savename='bands'):
    img = spectral.envi.open(hdr_path)
    print(img)
    print("shape:", img.shape)
    img_open = img.open_memmap(writeable=True)  # todo set writeable = False
    print("plot bands...")
    figure, axis = plt.subplots(1, 4, figsize=(24, 18))

    r_band = img_open[:, :, bands[0]]
    axis[0].imshow(r_band, cmap='viridis')
    axis[0].set_title('band no. ' + str(bands[0]))
    g_band = img_open[:, :, bands[1]]
    axis[1].imshow(g_band, cmap='viridis')
    axis[1].set_title('band no. ' + str(bands[1]))
    b_band = img_open[:, :, bands[2]]
    axis[2].imshow(b_band, cmap='viridis')
    axis[2].set_title('band no. ' + str(bands[2]))
    rgb_norm = create_rgb_norm((r_band, g_band, b_band))
    axis[3].imshow(rgb_norm, cmap='viridis')
    axis[3].set_title('combined')
    if save:
        print("save...")
        plt.savefig('../output/UFZ_flightdata/' + str(savename) + '_' + str(bands) + '.png', bbox_inches='tight')
    plt.show()


def plot_bands_rasterio(origin, bands):
    rgb = get_bands_from_raster(origin, bands)
    rgb = (rgb[0] * 0.75, rgb[1] * 1.05, rgb[2] * 1.1)
    plot_3_band_image_clipped(rgb, title='', cmap='viridis', max_reflectance = 2000)


FILE_PATH = '../../data/UFZ_flightdata/leipzig-auwald-sued_20230612_ref_geo_mosaic.bsq'
HEADER_PATH = '../../data/UFZ_flightdata/leipzig-auwald-sued_20230612_ref_geo_mosaic.hdr'
# HDR cheat sheet: https://www.nv5geospatialsoftware.com/docs/ENVIHeaderFiles.html

# rgb_bands = [120, 70, 40]
# rgb_bands = [200, 300, 400]
rgb_bands = [92, 43, 9]
# plot_bands(HEADER_PATH, rgb_bands, save=False, savename='auwald_sued')
# works also (and maybe more easily) with rasterio:
raster = rasterio.open(FILE_PATH)
# plot_bands_rasterio(raster, rgb_bands)
print(raster.width, raster.height)


# vegetation indices test:
# img = spectral.envi.open(HEADER_PATH)
# meta = img.metadata
# print('x-y-start coordinates: ' + str(meta['map info'][3]) + ' | ' + str(meta['map info'][4]))
# print('spatial resolution: ' + str(meta['map info'][5]) + ' x ' +
#       str(meta['map info'][6]) + ' | ' + meta['map info'][-1])
# vi = spectral.ndvi(img, 200, 120)
# plt.imshow(vi)
# plt.show()
