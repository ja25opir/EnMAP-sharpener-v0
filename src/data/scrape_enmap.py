import sys, os, json
import requests


def download_file(session, url, save_dir, file_format='tif'):
    response = session.get(url)
    if response.status_code != 200:
        print('Error downloading file. Status code:', response.status_code)
        sys.exit()
    tif_bytes = response.content
    save_name = url.split('/')[-1].split('.')[0].strip('_COG')
    with open(save_dir + save_name + '.' + file_format, 'wb') as f:
        f.write(tif_bytes)


def get_item_list(session, url, start_index=None):
    parameter = {
        'f': 'json',
        'limit': 20,
        'startIndex': start_index,
    }
    enmap_collection_response = session.get(url, params=parameter)
    print(enmap_collection_response.url)
    collection = enmap_collection_response.content.decode('utf-8')
    if enmap_collection_response.status_code == 200:
        print('Successfully retrieved item list.')
        return json.loads(collection)
    else:
        print('Error retrieving item list. Status code:', enmap_collection_response.status_code)
        sys.exit()


class EnMAP:
    def __init__(self, enmap_dir, max_cloud_cover, start_index, session_token):
        self.enmap_dir = enmap_dir
        self.max_cloud_cover = max_cloud_cover
        self.start_index = start_index
        self.session_token = session_token
        self.auth_session = set_auth(self)
        self.default_index_url = 'https://geoservice.dlr.de/eoc/ogc/stac/v1/collections/ENMAP_HSI_L2A/items'
        self.downloaded_scenes = 0
        self.checked_scenes = 0

    def set_auth(self):
        # load_dotenv(dotenv_path=find_dotenv())
        # username = os.getenv('GEOSERVICE_DLR_USERNAME')
        # password = os.getenv('GEOSERVICE_DLR_PASSWORD')
        # form_data = {
        #     'username': username,
        #     'password': password,
        # }
        cookie = 'session=' + self.session_token
        session = requests.session()
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Cookie': cookie,
            'Host': 'download.geoservice.dlr.de',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 OPR/107.0.0.0'
        }
        session.headers.update(headers)
        return session

    def filter_item_list(self, session, item_feature_list, max_cloud_cover, number_all_scenes):
        for item in item_feature_list:
            print('Downloading item', self.checked_scenes + 1, 'of', number_all_scenes, '...')
            cloud_cover = int(item['properties']['eo:cloud_cover']) + int(item['properties']['enmap:cirrus_cover'])
            if cloud_cover < max_cloud_cover and item['properties'][
                'enmap:sceneAOT'] != '-999':
                metadata_href = item['assets']['metadata']['href']
                scene_dir = self.enmap_dir + metadata_href.split('/')[-1].split('-META')[0] + '/'
                os.mkdir(scene_dir)
                download_file(session, metadata_href, scene_dir, 'xml')
                spectral_image_href = item['assets']['image']['href']
                download_file(session, spectral_image_href, scene_dir)
                quality_cloud_href = item['assets']['quality_cloud']['href']
                download_file(session, quality_cloud_href, scene_dir)
                self.downloaded_scenes += 1
                print('Done!')
            else:
                print('Cloud cover too high or broken file detected! Skipping this scene.')
            self.checked_scenes += 1

    def scrape_all_scenes(self):
        next_link = self.default_index_url
        while next_link != '':

            item_list = get_item_list(self.auth_session, next_link, self.start_index)
            self.start_index = None
            scenes = item_list['features']
            self.filter_item_list(self.auth_session, scenes, self.max_cloud_cover, item_list['numberMatched'])
            for link in item_list['links']:
                if link['rel'] == 'next':
                    next_link = link['href']
                    break
                else:
                    next_link = ''
