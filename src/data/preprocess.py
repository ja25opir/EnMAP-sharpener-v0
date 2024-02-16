import pycrs
import rasterio.mask


def crop_raster(raster, shape):
    print("crop raster...")
    print(raster.profile)
    print(raster.bounds)

    out_img, out_transform = rasterio.mask.mask(raster, shapes=shape, crop=True)
    # update metadata
    out_meta = raster.meta.copy()
    epsg_code = int(raster.crs.data['init'][5:])
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform,
                     "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()
                     })
    print(out_meta)
    return out_img, out_meta
