import os, re
import rasterio
import time
from rasterio.io import MemoryFile
import numpy as np


def resample_raster(raster, resample_size):
    return raster.read(
        out_shape=(raster.count,
                   resample_size[0],
                   resample_size[1]),
        resampling=rasterio.enums.Resampling.bilinear
    )


def resample_raster_in_memory(input_raster, resample_size):
    new_width = resample_size[1]
    new_height = resample_size[0]

    # create the transformation matrix and meta for the new raster
    out_transform = input_raster.transform * input_raster.transform.scale(
        (input_raster.width / new_width),
        (input_raster.height / new_height)
    )
    meta = input_raster.meta.copy()
    meta.update({"driver": "GTiff",
                 "height": new_height,
                 "width": new_width,
                 "transform": out_transform,
                 })

    # create an in-memory file
    with MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            # resample the data using bilinear interpolation
            data = input_raster.read(
                out_shape=(input_raster.count, new_height, new_width),
                resampling=rasterio.enums.Resampling.bilinear
            )

            # write the rescaled data to the in-memory dataset
            dataset.write(data)

        # return the rescaled raster as a rasterio dataset object
        return memfile.open()


def stack_rasters(raster1, raster2):
    raster_stack = []
    for band in range(1, raster1.count + 1):
        raster_stack.append(raster1.read(band))
    for band in range(1, raster2.count + 1):
        raster_stack.append(raster2.read(band))

    meta = raster1.meta.copy()
    meta.update(count=len(raster_stack))
    with MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            dataset.write(np.array(raster_stack))
        return memfile.open()


def tile_raster(raster, tile_size, save_dir, save_name, min_value_ratio=0.3, overlap=0):
    """
    Slices a raster into tiles of size tile_size x tile_size and saves them as .npy files.
    Tiles without any values or with fewer values than a given ratio are skipped.
    :param raster:
    :param tile_size:
    :param save_dir:
    :param save_name:
    :param min_value_ratio:
    :param overlap:
    :return:
    """
    horizontal_tiles = int(raster.width / (tile_size - overlap))
    vertical_tiles = int(raster.height / (tile_size - overlap))
    skip_list = []
    minimum_values = min_value_ratio * tile_size * tile_size * raster.count
    for i_h in range(horizontal_tiles):
        for i_v in range(vertical_tiles):
            file_name = f'{save_name}_{i_h}_{i_v}.npy'
            w = raster.read(window=((i_v * (tile_size - overlap), (i_v + 1) * tile_size - i_v * overlap),
                                    (i_h * (tile_size - overlap), (i_h + 1) * tile_size - i_h * overlap)))
            if not np.any(w):
                skip_list.append(file_name)
                continue
            elif np.count_nonzero(w) < minimum_values:
                skip_list.append(file_name)
                continue
            else:
                np.save(f'{save_dir}{file_name}', w)
    return skip_list

def start_wald_protocol(dir_path, enmap_file, sentinel_file, save_name, output_dir_path):
    enmap_raster = rasterio.open(dir_path + enmap_file)
    sentinel_raster = rasterio.open(dir_path + sentinel_file)

    print('Resampling...')
    start_time = time.time()
    enmap_downscaled = resample_raster_in_memory(enmap_raster,
                                                 (int(enmap_raster.height / 3), int(enmap_raster.width / 3)))
    sentinel_downscaled = resample_raster_in_memory(sentinel_raster,
                                                    (int(sentinel_raster.height / 3), int(sentinel_raster.width / 3)))
    enmap_rescaled = resample_raster_in_memory(enmap_downscaled, sentinel_downscaled.shape)
    print("Resampling time: %.4fs" % (time.time() - start_time))

    print('Stacking resampled EnMAP and downsampled Sentinel rasters...')
    start_time = time.time()
    x_image = stack_rasters(enmap_rescaled, sentinel_downscaled)
    print("Stacking time: %.4fs" % (time.time() - start_time))

    print('Tiling and saving X and Y image...')
    start_time = time.time()
    x_tiles_path = output_dir_path + 'x/'
    sparse_x_tiles = tile_raster(x_image, 100, x_tiles_path, save_name, min_value_ratio=0.3, overlap=0)
    y_tiles_path = output_dir_path + 'y/'
    sparse_y_tiles = tile_raster(enmap_raster, 100, y_tiles_path, save_name, min_value_ratio=0.3, overlap=0)

    # remove partner tiles if one of the pair was skipped
    for file in sparse_x_tiles:
        if os.path.exists(y_tiles_path + file):
            os.remove(y_tiles_path + file)
    for file in sparse_y_tiles:
        if os.path.exists(x_tiles_path + file):
            os.remove(x_tiles_path + file)

    print("Tiling time: %.4fs" % (time.time() - start_time))
