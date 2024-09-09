import os

import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
import resource

def limit_logical_cpus(logical_cpus):
    """Set process CPU affinity to the first max_cpus CPUs"""
    os.sched_setaffinity(os.getpid(), logical_cpus)
    print('Using following CPUs: ', os.sched_getaffinity(os.getpid()))


def limit_memory_usage(max_memory_limit_gb):
    """Set process memory limit to max_memory_limit_gb GB"""
    max_memory_limit_bytes = max_memory_limit_gb * 1024 * 1024 * 1024  # GB to bytes
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_limit_bytes, max_memory_limit_bytes))  # soft and hard limit
    print('Using following memory limit: ', resource.getrlimit(resource.RLIMIT_AS)[1] / 1024 / 1024 / 1024, 'GB')


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


def resample_raster(raster_name, raster, resample_size, save_name, output_dir):
    print('Raster loaded! Reading into memory and downsampling...')
    out_img = raster.read(
        out_shape=(raster.count,
                   resample_size[0],
                   resample_size[1]),
        resampling=rasterio.enums.Resampling.nearest
    )

    print('Raster resampled! Saving...')
    # scale image transform
    out_transform = raster.transform * raster.transform.scale(
        (raster.width / out_img.shape[-1]),
        (raster.height / out_img.shape[-2])
    )
    out_meta = raster.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform,
                     })
    with rasterio.open(output_dir + raster_name.strip('.tif') + save_name, "w",
                       **out_meta) as dest:
        dest.write(out_img)


FILE_PATH = os.getcwd() + '/../20230612_leipzig-auwald-s_final-mosaic/leipzig-auwald-sued_20230612_ref_geo_mosaic.bsq'
# HEADER_PATH = '../../data/UFZ_flightdata/leipzig-auwald-sued_20230612_ref_geo_mosaic.hdr'

raster = rasterio.open(FILE_PATH)

margin_y = 200
margin_x = 200
ul = raster.bounds[0] - margin_x, raster.bounds[3] + margin_y
ur = raster.bounds[2] - margin_x, raster.bounds[3] + margin_y
lr = raster.bounds[2] - margin_x, raster.bounds[1] + margin_y
ll = raster.bounds[0] - margin_x, raster.bounds[1] + margin_y
bbox = [[ul[0], ur[1]], [lr[0], ur[1]], [lr[0], ll[1]], [ul[0], ll[1]], [ul[0], ur[1]]]
crop_shape = [{'type': 'Polygon',
               'coordinates': [bbox]}]
UFZ_path = os.getcwd() + '/data/UFZ_flightdata/'
print('Cropping raster...')
# cropped_raster, out_meta = crop_raster(raster, crop_shape, mem_raster=True, save=True, output_dir=UFZ_path, save_name='UFZ_cropped_raster')
print('Raster cropped!')

limit_logical_cpus([0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
limit_memory_usage(50)

UFZ_cropped = os.getcwd() + '/data/UFZ_flightdata/UFZ_cropped_raster.tif'
resample_raster('UFZ_cropped_raster.tif', rasterio.open(UFZ_cropped), (287, 210), '_resampled.tif',
                os.getcwd() + '/data/UFZ_flightdata/')
