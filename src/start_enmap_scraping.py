import os, argparse

from data.scrape_enmap import EnMAP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start scraping EnMAP scenes from geoservice.dlr.de')
    parser.add_argument('--enmap-dir', type=str, default='/data/EnMAP_scrape/', help='Path to the directory where the EnMAP data will be saved')
    parser.add_argument('--max-cloud-cover', type=str, default='70', help='Maximum cloud cover percentage')

    args = parser.parse_args()

    enmap_dir_path = os.getcwd() + args.enmap_dir
    enmap = EnMAP(enmap_dir_path, args.max_cloud_cover)
    enmap.scrape_all_scenes()
