import os, argparse, time

from data.scrape_enmap import EnMAP
from config.resource_limiter import limit_logical_cpus, limit_memory_usage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start scraping EnMAP scenes from geoservice.dlr.de')
    parser.add_argument('--enmap-dir', type=str, default='/data/EnMAP/',
                        help='Path to the directory where the EnMAP data will be saved')
    parser.add_argument('--max-cloud-cover', type=int, default=70, help='Maximum cloud and cirrus cover in percentage')
    parser.add_argument('--bbox', nargs='+', type=float, default=[], help='Bounding box for scene acquisition')
    parser.add_argument('--date-time', type=str, default=None, help='Interval for scene acquisition in ISO 8601 format')
    parser.add_argument('--start-index', type=str, default='0', help='Index of first scene to download')
    parser.add_argument('--max-scenes', type=int, default=1000, help='Maximum number of scenes to download')
    parser.add_argument('--timestamps-file', type=int, default=0,
                        help='Controls if a file with timestamps from all matching scenes is created. '
                             '0: no file, 1: file with timestamps only (no scene download), 2: file with timestamps and scene download')
    parser.add_argument('--session-token', type=str, default='', help='Session token for DLR authentication')
    parser.add_argument('--cpus', nargs='+', type=int, default=[0, 1, 2, 3],
                        help='Assigned logical CPUs for the pipeline')
    parser.add_argument('--mem-limit', type=int, default=1, help='Memory limit for the pipeline in GB')

    args = parser.parse_args()

    if args.max_cloud_cover < 0 or args.max_cloud_cover > 100:
        raise ValueError('Cloud cover must be a percentage between 0 and 100.')

    limit_logical_cpus(args.cpus)
    limit_memory_usage(args.mem_limit)

    start_time = time.time()
    cpu_start = time.process_time()

    enmap_dir_path = os.getcwd() + args.enmap_dir
    enmap = EnMAP(enmap_dir_path, args.max_cloud_cover, args.bbox, args.date_time, args.start_index, args.max_scenes,
                  args.timestamps_file, args.session_token)
    enmap.scrape_all_scenes()

    print("---EnMAPScraping---Elapsed time: %.2fs seconds ---" % (time.time() - start_time))
    print("---EnMAPScraping---CPU time: %.2fs seconds ---" % (time.process_time() - cpu_start))
