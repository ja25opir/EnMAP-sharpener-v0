import os

from data.scrape_enmap import EnMAP

if __name__ == '__main__':
    ENMAP_DIR = os.getcwd() + '/data/EnMAP_scrape/'
    MAX_CLOUD_COVER = '70'  # %
    enmap = EnMAP(ENMAP_DIR, MAX_CLOUD_COVER)
    enmap.scrape_all_scenes()
