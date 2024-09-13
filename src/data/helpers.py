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