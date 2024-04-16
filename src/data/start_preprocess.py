import argparse, time
import os

from preprocess_pipeline import PreprocessPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start pipeline to preprocess data for model training.')
    parser.add_argument('--enmap-dir', type=str, default='/data/EnMAP/', help='Path to the directory with EnMAP data')
    parser.add_argument('--output-dir', type=str, default='/data/preprocessing/',
                        help='Path to the directory where the output data will be saved')
    parser.add_argument('--stages', nargs='+', default='[all]', help='Stages to run (all, crop, scrape, mask, wald')

    args = parser.parse_args()

    start_time = time.time()
    cpu_start = time.process_time()

    enmap_dir_path = os.getcwd() + args.enmap_dir
    output_dir_path = os.getcwd() + args.output_dir
    pipeline = PreprocessPipeline(enmap_dir_path, output_dir_path)

    if 'all' in args.stages:
        pipeline.start_all_steps()
    elif 'crop' in args.stages:
        pipeline.crop_all()
    elif 'scrape' in args.stages:
        pipeline.scrape_all()
    elif 'mask' in args.stages:
        pipeline.cloud_mask_all()
    elif 'wald' in args.stages:
        pipeline.wald_protocol_all()
    elif args.stages == 'scrape':
        pipeline.scrape_all()
    elif args.stages == 'cloud_mask':
        pipeline.cloud_mask_all()
    elif args.stages == 'wald_protocol':
        pipeline.wald_protocol_all()

    print("---PreprocessPipeline---Elapsed time: %.2fs seconds ---" % (time.time() - start_time))
    print("---PreprocessPipeline---CPU time: %.2fs seconds ---" % (time.process_time() - cpu_start))
