import os
import rasterio
import time
import numpy as np
import cv2
from rasterio.io import MemoryFile
from rasterio.windows import Window

from .helpers import crop_raster


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
                # resampling=rasterio.enums.Resampling.cubic
            )

            # write the rescaled data to the in-memory dataset
            dataset.write(data)

        # return the rescaled raster as a rasterio dataset object
        return memfile.open()


def get_gradient(img):
    # calculate x and y gradients using Sobel operator
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    # combine gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def align_sentinel(enmap_raster, sentinel_raster_downscaled):
    """align Sentinel Raster to EnMAP Raster with ECC algorithm and openCV"""
    # read 4 enmap bands near to sentinel bands as array
    enmap_array = enmap_raster.read((15, 29, 47, 71)).T.astype(np.uint8)
    # downscale sentinel to enmap shape and read as array
    sentinel_array = sentinel_raster_downscaled.read((1, 2, 3, 4)).T.astype(np.uint8)

    # ECC algorithm parameters
    warp_mode = cv2.MOTION_TRANSLATION  # translation only (no rotation or scaling)
    warp_mat = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 5000
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    warp_matrices = []
    # find warp matrices with ECC and gradients of four bands
    for band in range(4):
        e_gradient = get_gradient(enmap_array[:, :, band])
        s_gradient = get_gradient(sentinel_array[:, :, band])

        (cc, warp_mat) = cv2.findTransformECC(e_gradient, s_gradient, warp_mat, warp_mode, criteria)
        # only use integer values for translation (no interpolation)
        if not len(warp_matrices):
            warp_matrices = [warp_mat]
        else:
            warp_matrices.append(warp_mat)

    # average matrix of all gradient warp matrices
    sum_matrix = np.sum(warp_matrices, axis=0)
    warp_matrix = sum_matrix / len(warp_matrices)

    # round to integer values to avoid interpolation
    warp_matrix = np.round(warp_matrix).astype(np.float32)

    # apply affine transformation to original sentinel image
    sentinel_original = sentinel_raster_downscaled.read((1, 2, 3, 4)).T
    sentinel_aligned = cv2.warpAffine(sentinel_original, warp_matrix, (enmap_array.shape[1], enmap_array.shape[0]),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return sentinel_aligned.T, {'y': warp_matrix[0, -1], 'x': warp_matrix[1, -1]}


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


def stack_raster_and_array(raster1, array2):
    raster_stack = []
    for band in range(1, raster1.count + 1):
        raster_stack.append(raster1.read(band))
    for band in range(array2.shape[0]):
        raster_stack.append(array2[band, :, :])

    meta = raster1.meta.copy()
    meta.update(count=len(raster_stack))
    with MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            dataset.write(np.array(raster_stack))
        return memfile.open()


def crop_after_warp(raster, warp_dict):
    """crop raster after warp to remove borders caused by translation"""
    resolution = raster.transform[0]
    x_shift = warp_dict['x'] * resolution
    y_shift = warp_dict['y'] * resolution
    # construct bbox coordinates
    ul = [raster.bounds.left, raster.bounds.top]
    ur = [raster.bounds.right, raster.bounds.top]
    lr = [raster.bounds.right, raster.bounds.bottom]
    ll = [raster.bounds.left, raster.bounds.bottom]
    # shift bbox coordinates from warp_dict
    if x_shift < 0:
        ul[0] -= x_shift
        ll[0] -= x_shift
    else:
        ur[0] -= x_shift
        lr[0] -= x_shift
    if y_shift < 0:
        ul[1] -= y_shift
        ur[1] -= y_shift
    else:
        ur[1] -= y_shift
        lr[1] -= y_shift
    bbox = [ul, ur, lr, ll, ul]
    # crop from new bbox
    crop_shape = [{'type': 'Polygon', 'coordinates': [bbox]}]
    (raster_cropped, meta) = crop_raster(raster, crop_shape)

    # create raster from array and write into memory
    with MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            dataset.write(raster_cropped)
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
    :return: list with filenames of skipped tiles
    """
    # set a margin as pixel values on the left and upper border of enmap scenes are sometimes skewed
    left_edge_margin = 5
    top_edge_margin = 5
    horizontal_tiles = int(raster.width / (tile_size - overlap + top_edge_margin))
    vertical_tiles = int(raster.height / (tile_size - overlap + left_edge_margin))
    skip_list = []
    minimum_values = min_value_ratio * tile_size * tile_size * raster.count

    for i_h in range(horizontal_tiles):
        for i_v in range(vertical_tiles):
            file_name = f'{save_name}_{i_h}_{i_v}.npy'
            left_x = i_v * (tile_size - overlap) + left_edge_margin
            left_y = i_h * (tile_size - overlap)
            w = raster.read(window=Window(left_x, left_y, tile_size, tile_size))
            if not np.any(w):
                skip_list.append(file_name)
                continue
            elif np.count_nonzero(w) < minimum_values:
                skip_list.append(file_name)
                continue
            else:
                np.save(f'{save_dir}{file_name}', w)
    return skip_list


def remove_tile(path, name):
    if os.path.exists(path + name):
        os.remove(path + name)


def start_wald_protocol(dir_path, tile_size, enmap_file, sentinel_file, save_name, output_dir_path,
                        save_lr_enmap=False):
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

    print('Aligning Sentinel raster to EnMAP raster...')
    start_time = time.time()
    (sentinel_aligned, warp_dictionary) = align_sentinel(enmap_rescaled, sentinel_downscaled)
    print("Alignment time: %.4fs" % (time.time() - start_time))

    print('Stacking resampled EnMAP and downsampled Sentinel rasters...')
    start_time = time.time()
    x_image = stack_raster_and_array(enmap_rescaled, sentinel_aligned)
    x_image = crop_after_warp(x_image, warp_dictionary)
    print("Stacking time: %.4fs" % (time.time() - start_time))

    print('Tiling and saving X and Y image...')
    start_time = time.time()
    min_value_ratio = 0.9  # minimum ratio of non-zero values in a tile; others are discarded
    # x = stacked resampled raster, y = original EnMAP raster, x1 = resampled EnMAP raster
    x_tiles_path = output_dir_path + 'x/'
    sparse_x_tiles = tile_raster(x_image, tile_size, x_tiles_path, save_name, min_value_ratio=min_value_ratio,
                                 overlap=0)
    y_tiles_path = output_dir_path + 'y/'
    sparse_y_tiles = tile_raster(enmap_raster, tile_size, y_tiles_path, save_name, min_value_ratio=min_value_ratio,
                                 overlap=0)

    # remove partner tiles if one of the pair was skipped
    for file in sparse_x_tiles:
        remove_tile(y_tiles_path, file)
    for file in sparse_y_tiles:
        remove_tile(x_tiles_path, file)

    # save resampled EnMAP raster as x1 files # todo: maybe not even necessary, can be handled in DataLoader
    if save_lr_enmap:
        x1_tiles_path = output_dir_path + 'x1/'
        sparse_x1_tiles = tile_raster(enmap_rescaled, tile_size, x1_tiles_path, save_name,
                                      min_value_ratio=min_value_ratio, overlap=0)
        for file in sparse_x1_tiles:
            remove_tile(y_tiles_path, file)
            remove_tile(x_tiles_path, file)
        for file in sparse_x_tiles:
            remove_tile(x1_tiles_path, file)
        for file in sparse_y_tiles:
            remove_tile(x1_tiles_path, file)

    print("Tiling time: %.4fs" % (time.time() - start_time))
