import numpy as np

def print_raster_info(raster):
    print('number of bands:', raster.count)
    print('raster shape:', raster.shape)
    print('no. cells (shape[0] x shape[1]):', np.prod(raster.shape))
    print('spatial expansion:', str(raster.shape[0] * raster.res[0]) + raster.crs.linear_units + ' x',
          str(raster.shape[1] * raster.res[1]) + raster.crs.linear_units)
    print('bounding coordinates:', raster.bounds)
    print('x-y-start coordinates: ' + str(raster.transform[2]) + ' | ' + str(raster.transform[5]))
    print('transformation matrix:\n' + str(raster.transform))
    print('spatial resolution: ' + str(raster.transform[0]) + ' x ' +
          str(-raster.transform[4]) + ' ' + raster.crs.linear_units)
    print('coordinate reference system:', raster.crs)


def get_bands_from_raster(raster, bands):
    # TODO: currently only for visualisation as it works with 3 given bands
    return raster.read(bands[0]), raster.read(bands[1]), raster.read(bands[2])


def get_bands_from_array(array, bands):
    # TODO: currently only for visualisation as it works with 3 given bands
    return array[bands[0], :, :], array[bands[1], :, :], array[bands[2], :, :]
