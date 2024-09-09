import os

import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile


def crop_raster(raster, shape, mem_raster=False, save=False, output_dir='', save_name=''):
    """
    Crops a raster to a given shape.
    :param mem_raster:
    :param raster:
    :param shape:
    :param save:
    :param output_dir:
    :param save_name:
    :return:
    """
    out_img, out_transform = mask(raster, shapes=shape, crop=True)
    # update metadata
    out_meta = raster.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform,
                     })

    if save:
        print('saving to:', output_dir + save_name + '.tif')
        with rasterio.open(output_dir + save_name + '.tif', "w", **out_meta) as dest:
            dest.write(out_img)

    if mem_raster:
        with MemoryFile() as memfile:
            with memfile.open(**out_meta) as dataset:
                dataset.write(out_img)
            # return the rescaled raster as a rasterio dataset object
            return memfile.open(), out_meta

    return out_img, out_meta


FILE_PATH = os.getcwd() + '/../20230612_leipzig-auwald-s_final-mosaic/leipzig-auwald-sued_20230612_ref_geo_mosaic.bsq'
# HEADER_PATH = '../../data/UFZ_flightdata/leipzig-auwald-sued_20230612_ref_geo_mosaic.hdr'

raster = rasterio.open(FILE_PATH)

margin_y = 0
margin_x = 0
ul = raster.bounds[0] - margin_x, raster.bounds[3] + margin_y
ur = raster.bounds[2] - margin_x, raster.bounds[3] + margin_y
lr = raster.bounds[2] - margin_x, raster.bounds[1] + margin_y
ll = raster.bounds[0] - margin_x, raster.bounds[1] + margin_y
bbox = [[ul[0], ur[1]], [lr[0], ur[1]], [lr[0], ll[1]], [ul[0], ll[1]], [ul[0], ur[1]]]
crop_shape = [{'type': 'Polygon',
               'coordinates': [bbox]}]
UFZ_path = os.getcwd() + '/data/UFZ_flightdata/'
cropped_raster, out_meta = crop_raster(raster, crop_shape, mem_raster=True, save=True, output_dir=UFZ_path, save_name='UFZ_cropped_raster')
