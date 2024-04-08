import os
from datetime import timedelta
import rasterio
from dateutil.parser import parse
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv, find_dotenv
import io
import tarfile


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


def unzip_response_files(response, output_dir, save_name):
    tar_bytes = response.content
    tar_stream = io.BytesIO(tar_bytes)
    with tarfile.open(fileobj=tar_stream) as tar:
        for member in tar.getmembers():
            original_filename = os.path.basename(member.name)
            new_filename = save_name + '_sentinel_' + original_filename
            tar.extract(member, path=output_dir, set_attrs=False)
            os.rename(os.path.join(output_dir, original_filename),
                      os.path.join(output_dir, new_filename))


def request_and_save_response(image_path, time, output_dir, save_name):
    print("Requesting Sentinel image for EnMAP scene: ", save_name + '...')
    sentinel = Sentinel(image_path, time)
    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    # request options: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L1C.html
    # response = OAUTH_SESSION.post(url, json=sentinel.request(), headers={"Accept": "image/tiff"})
    response = OAUTH_SESSION.post(url, json=sentinel.request(), headers={"Accept": "application/tar"})

    print('Response status:', response.status_code)
    if response.status_code == 200:
        unzip_response_files(response, output_dir, save_name)
    else:
        print(response.content)


OAUTH_SESSION = get_auth_token()

# ProcessAPI examples https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Process/Examples/S2L2A.html
# OpenAPI: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/ApiReference.html#tag/process
# further explanation: https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html

# bands: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
# SCL: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/processing-levels/level-2#:~:text=The%20Scene%20Classification%20(SCL)%20algorithm,vegetation%2C%20not%20vegetated%2C%20water%20and
# CLM/CLP: https://docs.sentinel-hub.com/api/latest/user-guides/cloud-masks/
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ 
      bands: ["B02", "B03", "B04", "B08", "SCL"], // bands to be used + scene classification algorithm for masking clouds
      units: "DN" // value = reflectance * 10000
    }],
    output: [
    { 
      id: "spectral",
      bands: 4,
      sampleType: "INT16" // output type
    },{ 
      id: "cloud_mask",
      bands: 1,
      sampleType: "AUTO" // output type
    },]
  }
}

// masks cloudy pixels
function evaluatePixel(sample) {
  let cloudmask = 0
  if ([8, 9, 10].includes(sample.SCL)) {
      cloudmask = 1
  }
  return {
    spectral: [sample.B02, sample.B03, sample.B04, sample.B08], 
    cloud_mask: [cloudmask] // cloud mask
  }
}
"""


class Sentinel:
    def __init__(self, img_path, enmap_time, timerange_days=15):
        self.img_path = img_path
        self.enmap_raster = rasterio.open(img_path)
        self.bbox = [self.enmap_raster.bounds[0], self.enmap_raster.bounds[1], self.enmap_raster.bounds[2],
                     self.enmap_raster.bounds[3]]
        self.epsg_crs = self.enmap_raster.crs.to_epsg()
        self.time_from = None
        self.time_to = None
        self.timerange_days = timerange_days
        self.parse_time(enmap_time)
        self.evalscript = EVALSCRIPT

    def parse_time(self, time):
        timestamp = parse(time)
        time_from = timestamp - timedelta(days=self.timerange_days)
        time_to = timestamp + timedelta(days=self.timerange_days)
        self.time_from = time_from.strftime('%Y-%m-%dT%H:%M:%SZ')
        self.time_to = time_to.strftime('%Y-%m-%dT%H:%M:%SZ')

    def print_raster_info(self):
        print("Raster info:")
        print("Bounds:", self.enmap_raster.bounds)
        print("Width:", self.enmap_raster.width)
        print("Height:", self.enmap_raster.height)
        print("Crs:", self.enmap_raster.crs)
        print("Transform:", self.enmap_raster.transform)
        print("Count:", self.enmap_raster.count)
        print("Indexes:", self.enmap_raster.indexes)

    def request(self):
        print("bbox:", self.bbox)
        print("timerange:", self.time_from, " - ", self.time_to)
        return {
            "input": {
                "bounds": {
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/" + str(self.epsg_crs)},
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
                            "mosaickingOrder": "leastCC",  # select image from timerange with the least cloud coverage
                            # "maxCloudCoverage": 5,
                        },
                    }
                ],
            },
            # output cannot exceed 2500x2500 pixels which also limits the resolution
            # (e.g. 25.000m x 25.000m land section for 10m/pixel resolution)
            "output": {
                "resx": 10,
                "resy": 10,
                "responses": [
                    {
                        "identifier": "spectral",
                        "format": {"type": "image/tiff"}
                    },
                    {
                        "identifier": "cloud_mask",
                        "format": {"type": "image/tiff"}
                    }
                ]
            },
            "evalscript": self.evalscript,
        }


# TODO: (optional) validate found images
