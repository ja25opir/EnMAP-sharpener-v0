import argparse, os
from model.use_model import Reconstructor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start EnMAP sharpening.')
    parser.add_argument('--predictions-dir', type=str, default='/output/predictions/')
    parser.add_argument('--meta-path', type=str, default='/data/preprocessing/prediction_input/')
    parser.add_argument('--output-dir', type=str, default='/output/predictions/reconstructions/')
    args = parser.parse_args()

    predictions_dir = os.getcwd() + args.predictions_dir
    meta_dir = os.getcwd() + args.meta_path
    output_dir = os.getcwd() + args.output_dir

    # reconstruction
    reconstructor = Reconstructor(predictions_dir, meta_dir, output_dir)
    reconstructor.reconstruct_all()
