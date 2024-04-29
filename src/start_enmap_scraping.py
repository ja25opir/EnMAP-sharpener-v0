import os, argparse, time

from data.scrape_enmap import EnMAP
from config.resource_limiter import limit_logical_cpus, limit_memory_usage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start scraping EnMAP scenes from geoservice.dlr.de')
    parser.add_argument('--enmap-dir', type=str, default='/data/EnMAP_scrape/',
                        help='Path to the directory where the EnMAP data will be saved')
    parser.add_argument('--max-cloud-cover', type=str, default='70', help='Maximum cloud cover percentage')
    parser.add_argument('--cpus', nargs='+', type=int, default=[0, 1, 2, 3],
                        help='Assigned logical CPUs for the pipeline')
    parser.add_argument('--mem-limit', type=int, default=1, help='Memory limit for the pipeline in GB')

    args = parser.parse_args()

    limit_logical_cpus(args.cpus)
    limit_memory_usage(args.mem_limit)

    start_time = time.time()
    cpu_start = time.process_time()

    enmap_dir_path = os.getcwd() + args.enmap_dir
    enmap = EnMAP(enmap_dir_path, args.max_cloud_cover)
    enmap.scrape_all_scenes()

    print("---EnMAPScraping---Elapsed time: %.2fs seconds ---" % (time.time() - start_time))
    print("---EnMAPScraping---CPU time: %.2fs seconds ---" % (time.process_time() - cpu_start))
