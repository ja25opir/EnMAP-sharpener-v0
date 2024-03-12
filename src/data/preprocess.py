import rasterio.mask
from pyproj import Proj
import xml.etree.ElementTree as ETree
import os, re


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
    if width > max_width:
        ul[0] = int(ul[0] + (width - max_width) / 2) + 1  # +1 to avoid edge cases
        lr[0] = int(lr[0] - (width - max_width) / 2) - 1  # -1 to avoid edge cases
    if height > max_height:
        ur[1] = int(ur[1] - (height - max_height) / 2) - 1  # -1 to avoid edge cases
        ll[1] = int(ll[1] + (height - max_height) / 2) + 1  # +1 to avoid edge cases

    return [[ul[0], ur[1]], [lr[0], ur[1]], [lr[0], ll[1]], [ul[0], ll[1]], [ul[0], ur[1]]]


def crop_raster(raster, shape, save=False, output_dir='', savename=''):
    out_img, out_transform = rasterio.mask.mask(raster, shapes=shape, crop=True)
    # update metadata
    out_meta = raster.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform,
                     })

    if save:
        print('saving to:', output_dir + savename + '.tif')
        with rasterio.open(output_dir + savename + '.tif', "w", **out_meta) as dest:
            dest.write(out_img)

    return out_img, out_meta


def crop_enmap(metadata_path, spectral_img_path, output_dir):
    bbox = get_bounding_box_from_xml(metadata_path)
    origin = rasterio.open(spectral_img_path)
    ir_bbox = get_inscribed_rect_from_bbox(bbox, origin.crs)
    crop_shape = [{'type': 'Polygon',
                   'coordinates': [ir_bbox]}]
    timestamp = re.search('\d{4}\d{2}\d{2}T\d{6}Z', spectral_img_path)
    savename = timestamp.group() + '_enmap_cropped_spectral'
    origin_raster = rasterio.open(spectral_img_path)
    crop_raster(origin_raster, crop_shape, save=True, output_dir=output_dir, savename=savename)


class PreprocessPipeline:
    def __init__(self, enmap_dir_path, sentinel_dir_path):
        self.enmap_dir_path = enmap_dir_path
        self.enmap_subdir_suffix = 'EnMAP/ENMAP01.*'
        self.sentinel_dir_path = sentinel_dir_path
        pass

    def start_all_steps(self):
        self.crop_all()
        self.cloud_mask_all()
        self.scrape_all()


    def crop_all(self):
        print('Cropping EnMAP images... \n--------------------------')
        i = 0
        for directory in os.walk(self.enmap_dir_path):
            if re.search(self.enmap_subdir_suffix, directory[0]):
                metadata_path = ''
                spectral_img_path = ''
                for filename in directory[2]:
                    if re.search(".*METADATA.XML$", filename):
                        metadata_path = directory[0] + '/' + filename
                    if re.search(".*SPECTRAL_IMAGE.TIF$", filename):
                        spectral_img_path = directory[0] + '/' + filename
                i += 1
                if metadata_path and spectral_img_path:
                    print('Cropping image', i, 'of', len(os.listdir(self.enmap_dir_path)) - 1, '...')
                    output_dir = self.sentinel_dir_path
                    crop_enmap(metadata_path, spectral_img_path, output_dir)
                else:
                    print('No metadata or spectral image found in', directory[0])
        print('Cropping done.')

    def cloud_mask_all(self):
        pass

    def scrape_all(self):
        pass

    def wald_protocol(self):
        # ((maybe in model directory))
        # scale
        # combine cloud masks
        # (band co registering?)
        # tile
        pass

ENMAP_DIR_PATH = '../../data/EnMAP/'
OUTPUT_DIR = '../../data/Sentinel2/registered/'

pipeline = PreprocessPipeline(ENMAP_DIR_PATH, OUTPUT_DIR)
pipeline.crop_all()