import time
import rasterio.mask
from pyproj import Proj
import xml.etree.ElementTree as ETree
import os, re, sys
import numpy as np

from src.data.scrape_sentinel import request_and_save_response


def get_bounding_box_from_xml(xml_path):
    """
    Extracts bounding box coordinates for the spatial coverage from an EnMAP metadata XML file.
    .
    coordinate system: y increasing from bottom to top; x increasing from left to right
    [[x,y] upper left], [[x,y] upper right], [[x,y] lower right], [[x,y] lower left]
    .
    :param xml_path: path of the XML file
    :return: list of bounding box coordinates
    """
    root = ETree.parse(xml_path).getroot()
    bbox = []
    for base in root.findall('base'):
        for spatialCoverage in base.findall('spatialCoverage'):
            for boundingBox in spatialCoverage.findall('boundingPolygon'):
                for points in boundingBox.findall('point'):
                    if points[0].text != 'center':
                        lat = float(points[1].text)
                        lon = float(points[2].text)
                        bbox.append([lat, lon])
    return bbox


def long_lat_to_utm(lat, long, crs):
    """
    Convert longitude and latitude to UTM coordinates in one zone.
    Returns integer values.
    :param lat: latitude
    :param long: longitude
    :param crs: coordinate reference system
    :return: UTM coordinates
    """
    utm_zone = Proj(crs)
    x = int(utm_zone(long, lat)[0])
    y = int(utm_zone(long, lat)[1])
    return [x, y]


def get_inscribed_rect_from_bbox(bbox, origin_crs, max_width=25000, max_height=25000):
    """
    Calculate UTM coordinates for an axis parallel inscribed rectangle from a given bounding box consisting of 4 coordinates.
    Also crops the bounding box if it exceeds the maximum width or height.
    WARNING: only works for given rectangles rotated for != Z * 45°
    :param origin_crs: crs of the origin raster
    :param bbox: list of bounding box coordinates in lat/long
    :param max_width: maximum width of the inscribed rectangle in meters
    :param max_height: maximum height of the inscribed rectangle in meters
    :return: list of inscribed rectangle coordinates
    """
    ul = long_lat_to_utm(bbox[0][0], bbox[0][1], origin_crs)
    ll = long_lat_to_utm(bbox[1][0], bbox[1][1], origin_crs)
    lr = long_lat_to_utm(bbox[2][0], bbox[2][1], origin_crs)
    ur = long_lat_to_utm(bbox[3][0], bbox[3][1], origin_crs)
    width = lr[0] - ul[0]
    height = ur[1] - ll[1]
    # TODO: tweak margin
    margin = 100  # to avoid edge cases
    if width > max_width:
        ul[0] = int(ul[0] + (width - max_width) / 2) + int(margin / 2)
        lr[0] = int(lr[0] - (width - max_width) / 2) - int(margin / 2)
    if height > max_height:
        ur[1] = int(ur[1] - (height - max_height) / 2) - int(margin / 2)
        ll[1] = int(ll[1] + (height - max_height) / 2) + int(margin / 2)

    return [[ul[0], ur[1]], [lr[0], ur[1]], [lr[0], ll[1]], [ul[0], ll[1]], [ul[0], ur[1]]]


def crop_raster(raster, shape, save=False, output_dir='', save_name=''):
    """
    Crops a raster to a given shape.
    :param raster:
    :param shape:
    :param save:
    :param output_dir:
    :param save_name:
    :return:
    """
    out_img, out_transform = rasterio.mask.mask(raster, shapes=shape, crop=True)
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

    return out_img, out_meta


def crop_enmap(metadata_path, spectral_img_path, cloud_mask_path, output_dir):
    """
    Crops an EnMAP image and its cloud mask to the spatial coverage of the EnMAP metadata.
    :param metadata_path:
    :param spectral_img_path:
    :param cloud_mask_path:
    :param output_dir:
    :return:
    """
    bbox = get_bounding_box_from_xml(metadata_path)
    origin = rasterio.open(spectral_img_path)
    ir_bbox = get_inscribed_rect_from_bbox(bbox, origin.crs)
    crop_shape = [{'type': 'Polygon',
                   'coordinates': [ir_bbox]}]
    timestamp = re.search('\d{4}\d{2}\d{2}T\d{6}Z', spectral_img_path)
    save_name = timestamp.group() + '_enmap'
    origin_raster = rasterio.open(spectral_img_path)
    crop_raster(origin_raster, crop_shape, save=True, output_dir=output_dir, save_name=save_name + '_spectral')
    origin_cloud_raster = rasterio.open(cloud_mask_path)
    crop_raster(origin_cloud_raster, crop_shape, save=True, output_dir=output_dir, save_name=save_name + '_cloud_mask')


def get_time_from_enmap(enmap_path):
    time_match = re.search('\d{4}\d{2}\d{2}T\d{6}Z', enmap_path)
    try:
        enmap_time = time_match.group()
    except AttributeError:
        print("No timestamp found. Please provide a valid EnMAP image path. Exiting...")
        sys.exit(0)
    return enmap_time


def resample_raster(raster_name, raster, resample_size, save_name, output_dir):
    out_img = raster.read(
        out_shape=(raster.count,
                   resample_size[0],
                   resample_size[1]),
        resampling=rasterio.enums.Resampling.bilinear
    )
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


class PreprocessPipeline:
    def __init__(self, enmap_dir_path, output_dir_path):
        self.enmap_dir_path = enmap_dir_path
        self.enmap_subdir_suffix = 'ENMAP01.*'
        self.output_masks_path = output_dir_path + 'cloud_masks/'
        self.output_enmap_dir_path = output_dir_path + 'EnMAP/'
        self.output_sentinel_dir_path = output_dir_path + 'Sentinel2/'

    def start_all_steps(self):
        self.crop_all()
        self.scrape_all()
        self.cloud_mask_all()
        self.wald_protocol()

    def crop_all(self):
        print('Cropping EnMAP images... \n--------------------------')
        i = 0
        all_dirs = os.listdir(ENMAP_DIR_PATH)
        no_enmap_dirs = len([x for x in all_dirs if re.search(self.enmap_subdir_suffix, x)])
        for directory in os.walk(self.enmap_dir_path):
            if re.search(self.enmap_subdir_suffix, directory[0]):
                metadata_path = ''
                spectral_img_path = ''
                cloud_mask_path = ''
                for filename in directory[2]:
                    if re.search(".*METADATA.XML$", filename):
                        metadata_path = directory[0] + '/' + filename
                    if re.search(".*SPECTRAL_IMAGE.TIF$", filename):
                        spectral_img_path = directory[0] + '/' + filename
                    if re.search(".*QL_QUALITY_CLOUD.TIF$", filename):
                        cloud_mask_path = directory[0] + '/' + filename
                i += 1
                if metadata_path and spectral_img_path and cloud_mask_path:
                    print('Cropping image', i, 'of', no_enmap_dirs, '...')
                    output_dir = self.output_enmap_dir_path
                    crop_enmap(metadata_path, spectral_img_path, cloud_mask_path, output_dir)
                else:
                    print('No metadata or spectral image or cloud mask found in', directory[0])
        print('Cropping done.')

    def scrape_all(self):
        print('Scraping Sentinel images... \n--------------------------')
        for directory in os.walk(self.output_enmap_dir_path):
            for filename in directory[2]:
                if re.search(".*enmap_spectral.tif$", filename):
                    spectral_img_path = directory[0] + filename
                    time = get_time_from_enmap(spectral_img_path)
                    request_and_save_response(spectral_img_path, time, output_dir=self.output_sentinel_dir_path,
                                              save_name=time)

    # TODO: rename "sample" to "scale" (?)
    def cloud_mask_all(self):
        print('Upsampling and combining cloud masks... \n--------------------------')
        enmap_files = os.listdir(self.output_enmap_dir_path)
        enmap_cloud_masks = [x for x in enmap_files if re.search(".*cloud_mask.tif", x)]
        no_cloud_masks = len(enmap_cloud_masks)
        sentinel_files = os.listdir(self.output_sentinel_dir_path)
        sentinel_cloud_masks = [x for x in sentinel_files if re.search(".*cloud_mask.tif", x)]
        upscale_factor = 3
        i = 1
        for enmap_cloud_mask in enmap_cloud_masks:
            print('Upsampling cloud mask', i, 'of', no_cloud_masks, '...')
            enmap_cloud_raster = rasterio.open(self.output_enmap_dir_path + enmap_cloud_mask)
            upscaled_size = (enmap_cloud_raster.height * upscale_factor, enmap_cloud_raster.width * upscale_factor)
            resample_raster(enmap_cloud_mask, enmap_cloud_raster, upscaled_size, '_upsampled.tif',
                            self.output_enmap_dir_path)
            timestamp = enmap_cloud_mask.split('_')[0]

            for sentinel_cloud_mask in sentinel_cloud_masks:
                if re.search(timestamp, sentinel_cloud_mask):
                    print('Combining cloud masks...')
                    sentinel_cloud_raster = rasterio.open(self.output_sentinel_dir_path + sentinel_cloud_mask)
                    enmap_cloud_raster_upsampled = rasterio.open(
                        self.output_enmap_dir_path + enmap_cloud_mask.strip('.tif') + '_upsampled.tif')
                    meta = enmap_cloud_raster_upsampled.meta.copy()
                    merged_mask = np.logical_or(sentinel_cloud_raster.read(1), enmap_cloud_raster_upsampled.read(1))

                    with rasterio.open(self.output_masks_path + timestamp + '_cloud_mask_combined.tif', "w",
                                       **meta) as dest:
                        dest.write(merged_mask, indexes=1)

                    # mask sentinel & save
                    print('Masking Sentinel image...')
                    sentinel_raster = rasterio.open(
                        self.output_sentinel_dir_path + timestamp + '_sentinel_spectral.tif')
                    self.mask_raster(sentinel_raster, merged_mask, timestamp + '_sentinel_masked')

                    # downscale mask --> mask enmap & save
                    print('Downsampling mask & masking EnMaP image...')
                    enmap_raster = rasterio.open(self.output_enmap_dir_path + timestamp + '_enmap_spectral.tif')
                    merged_mask_raster = rasterio.open(self.output_masks_path + timestamp + '_cloud_mask_combined.tif')
                    resample_raster(timestamp + '_cloud_mask_combined.tif', merged_mask_raster, enmap_raster.shape,
                                    '_downsampled.tif', self.output_masks_path)
                    merged_mask_downsampled = rasterio.open(
                        self.output_masks_path + timestamp + '_cloud_mask_combined_downsampled.tif')
                    self.mask_raster(enmap_raster, merged_mask_downsampled, timestamp + '_enmap_masked')

                    print('Done!')
                    sentinel_cloud_masks.remove(sentinel_cloud_mask)
                    break
            i += 1

    def mask_raster(self, raster, mask, save_name):
        meta = raster.meta.copy()
        raster_np_stack = []

        for band in range(1, raster.count + 1):
            raster_np = np.array(raster.read(band))
            raster_np[mask == 1] = 0
            raster_np_stack.append(raster_np)

        with rasterio.open(self.output_masks_path + save_name + '.tif', 'w', **meta) as dst:
            for band_no, raster_band in enumerate(raster_np_stack, 1):
                dst.write(raster_band, band_no)

    def wald_protocol(self):
        # ((maybe in model directory))
        # scale --> cloud_mask_all
        # combine cloud masks --> cloud_mask_all
        # (band co registering?)
        # tile
        pass


ENMAP_DIR_PATH = '../../data/EnMAP/'
OUTPUT_DIR = '../../data/preprocessing/'

start_time = time.time()
cpu_start = time.process_time()

pipeline = PreprocessPipeline(ENMAP_DIR_PATH, OUTPUT_DIR)
pipeline.crop_all()
pipeline.scrape_all()
pipeline.cloud_mask_all()

print("---Elapsed time: %s seconds ---" % (time.time() - start_time))
print("---CPU time: %s seconds ---" % (time.process_time() - cpu_start))
