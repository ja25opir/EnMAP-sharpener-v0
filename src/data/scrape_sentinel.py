import os
import re
import sys
from datetime import timedelta
import rasterio
from dateutil.parser import parse
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv, find_dotenv


def get_auth_token():
    load_dotenv(dotenv_path=find_dotenv())

    # https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html#python
    CLIENT_ID = os.getenv('COPERNICUS_CLIENT_ID')
    CLIENT_SECRET = os.getenv('COPERNICUS_CLIENT_SECRET')

    # create a session
    client = BackendApplicationClient(client_id=CLIENT_ID)
    oauth = OAuth2Session(client=client)

    # get token for the session
    # all requests using this session will have an access token automatically added
    oauth.fetch_token(
        token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        client_secret=CLIENT_SECRET, include_client_id=True)

    return oauth


def request_save_response(image_path, time, evalscript, output_dir, save_name):
    print("Requesting Sentinel image for EnMAP scene: ", save_name + '...')
    sentinel = Sentinel(image_path, time, evalscript)
    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    # request options: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L1C.html
    response = OAUTH_SESSION.post(url, json=sentinel.request(), headers={"Accept": "image/tiff"})

    print('Response status:', response.status_code)
    file_name = 'Sentinel_' + save_name + '.tiff'
    with open(output_dir + file_name, "wb") as f:
        f.write(response.content)


class Sentinel:
    def __init__(self, img_path, enmap_time, evalscript, timerange_days=15):
        self.img_path = img_path
        self.enmap_raster = rasterio.open(img_path)
        self.bbox = [self.enmap_raster.bounds[0], self.enmap_raster.bounds[1], self.enmap_raster.bounds[2],
                     self.enmap_raster.bounds[3]]
        self.time_from = None
        self.time_to = None
        self.timerange_days = timerange_days
        self.parse_time(enmap_time)
        self.evalscript = evalscript

    def parse_time(self, time):
        timestamp = parse(time)
        time_from = timestamp - timedelta(days=self.timerange_days)
        time_to = timestamp + timedelta(days=self.timerange_days)
        self.time_from = time_from.strftime('%Y-%m-%dT%H:%M:%SZ')
        self.time_to = time_to.strftime('%Y-%m-%dT%H:%M:%SZ')

    def request(self):
        print("bbox:", self.bbox)
        print("timerange:", self.time_from, self.time_to)
        return {
            "input": {
                "bounds": {
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/32633"},  # EnMAP uses EPSG:32633
                    "bbox": self.bbox,
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": self.time_from,
                                "to": self.time_to,
                            },
                            "mosaickingOrder": "leastCC",  # select image from timerange with least cloud coverage
                            # "maxCloudCoverage": 5,
                        },
                    }
                ],
            },
            # "output": {
            #     "resx": 10,
            #     "resy": 10,
            # },
            "output": {
                "height": 500,
                "width": 500,
            },
            "evalscript": self.evalscript,
        }


# https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Process/Examples/S2L2A.html
# OpenAPI: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/ApiReference.html#tag/process

# bands: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B08", "SCL"], // bands to be used (select the ones we need)
    output: { bands: 4 },
  }
}

// masks cloudy pixels
function evaluatePixel(sample) {
  if ([8, 9, 10].includes(sample.SCL)) {
    return [1, 0, 0]
  } else {
    return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02]
  }
}
"""

EVALSCRIPT_DEFAULT = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "SCL"], // bands to be used (select the ones we need)
    output: { bands: 3 },
  }
}

function evaluatePixel(sample) {
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02]
}
"""

DIR_PATH = '../../data/EnMAP/'
OAUTH_SESSION = get_auth_token()
OUTPUT_DIR = '../../data/Sentinel2/scraped/'

# scrape sentinel images for a given directory containing EnMAP scenes
for directory in os.walk(DIR_PATH):
    if re.search("EnMAP/ENMAP01.*", directory[0]):
        for filename in directory[2]:
            if re.search(".*SPECTRAL_IMAGE.TIF$", filename):
                filepath = directory[0] + '/' + filename
                timeMatch = re.search('\d{4}\d{2}\d{2}T\d{6}Z', filename)
                try:
                    enmapTime = timeMatch.group()
                    request_save_response(filepath, enmapTime, EVALSCRIPT, OUTPUT_DIR, filename)
                except AttributeError:
                    print("No timestamp found. Please provide a valid EnMAP image path. Exiting...")
                    sys.exit(0)

# TODO: evalscript, output resolution, timerange, fileformat (currently tiff, maybe envi better?)
# TODO: validate found images
