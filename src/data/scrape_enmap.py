import sys, os, json, time, shutil
import requests


def download_file(session, url, save_dir, file_format='tif'):
    response = session.get(url, stream=True)
    save_name = url.split('/')[-1].split('.')[0].strip('_COG')
    if response.status_code == 200:
        chunk_size = 1024 * 1024 * 10  # 10 MB
        with open(save_dir + save_name + '.' + file_format, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        return response.status_code
    elif response.status_code == 404:
        return response.status_code
    else:
        print('Error downloading file. Status code:', response.status_code, '\n Filename:', save_name)
        sys.exit()


def get_item_list(session, url, parameter):
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
    def __init__(self, enmap_dir, max_cloud_cover, bbox, date_time, start_index, max_scenes, session_token):
        self.enmap_dir = enmap_dir
        self.max_cloud_cover = max_cloud_cover
        self.bbox = bbox
        self.date_time = date_time
        self.start_index = start_index
        self.max_scenes = max_scenes
        self.session_token = session_token
        self.auth_session = self.set_auth()
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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0'
        }
        session.headers.update(headers)
        return session

    def filter_item_list(self, session, item_feature_list, max_cloud_cover, number_all_scenes):
        for item in item_feature_list:
            print('Processing item', self.checked_scenes + 1 + int(self.start_index), 'of', number_all_scenes, '...')
            self.checked_scenes += 1
            cloud_cover = int(item['properties']['eo:cloud_cover']) + int(item['properties']['enmap:cirrus_cover'])
            if cloud_cover < max_cloud_cover and item['properties'][
                'enmap:sceneAOT'] != '-999':
                metadata_href = item['assets']['metadata']['href']
                scene_dir = self.enmap_dir + metadata_href.split('/')[-1].split('-META')[0] + '/'
                if os.path.exists(scene_dir):
                    print('Scene already downloaded. Skipping this scene.')
                    continue
                os.mkdir(scene_dir)

                # download spectral image, skip on 404, exit if token is invalid
                spectral_image_href = item['assets']['image']['href']
                start_time = time.time()
                status = download_file(session, spectral_image_href, scene_dir)
                if status == 404:
                    print('404 file not found. Skipping this scene.')
                    shutil.rmtree(scene_dir)
                    continue
                if (time.time() - start_time) < 3:
                    print('Session token expired or invalid. Please provide a new one.')
                    print('Failed item:', scene_dir)
                    shutil.rmtree(scene_dir)
                    sys.exit()
                # download metadata, skip on 404
                status = download_file(session, metadata_href, scene_dir, 'xml')
                if status == 404:
                    print('404 file not found. Skipping this scene.')
                    shutil.rmtree(scene_dir)
                    continue
                # download quality cloud, skip on 404
                quality_cloud_href = item['assets']['quality_cloud']['href']
                status = download_file(session, quality_cloud_href, scene_dir)
                if status == 404:
                    print('404 file not found. Skipping this scene.')
                    shutil.rmtree(scene_dir)
                    continue
                self.downloaded_scenes += 1
                print('Done!')
            else:
                print('Cloud cover too high or broken file detected! Skipping this scene.')
            if self.downloaded_scenes % 10 == 0:
                print('Downloaded', self.downloaded_scenes, 'scenes.')
            if self.downloaded_scenes >= self.max_scenes:
                print('Downloaded', self.max_scenes, 'scenes. Stopping download.')
                sys.exit()

    def scrape_all_scenes(self):
        next_link = self.default_index_url
        start_idx = self.start_index
        parameter = {
            'f': 'json',
            'limit': 20,
            'startIndex': start_idx,
            'bbox': self.bbox,
            'datetime': self.date_time
        }
        item_list = get_item_list(self.auth_session, next_link, parameter=parameter)
        scenes = item_list['features']
        # repeat as long as the pagination returns a next link
        while next_link != '':
            self.filter_item_list(self.auth_session, scenes, self.max_cloud_cover, item_list['numberMatched'])
            for link in item_list['links']:
                if link['rel'] == 'next':
                    next_link = link['href']
                    break
                else:
                    next_link = ''
            # initial parameter are provided in next_link (bbox gets cutoff so we provide the rest again)
            item_list = get_item_list(self.auth_session, next_link, parameter={'bbox': self.bbox[1:]})
            scenes = item_list['features']
