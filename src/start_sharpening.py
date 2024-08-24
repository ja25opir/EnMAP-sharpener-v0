import argparse, os
import tensorflow as tf
from model.architecture import ReflectionPadding2D, ReflectionPadding3D, SFTLayer, DILayer, ms_ssim_l1_loss
from model.use_model import Predictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start EnMAP sharpening.')
    parser.add_argument('--input-dir', type=str, default='/data/preprocessing/prediction_input/')
    parser.add_argument('--output-dir', type=str, default='/output/predictions/')
    parser.add_argument('--model', type=str, default='/output/models/supErMAPnet.keras')

    args = parser.parse_args()

    input_dir = os.getcwd() + args.input_dir
    output_dir = os.getcwd() + args.output_dir

    # load model with its custom layers
    CUSTOM_LAYERS = {'ReflectionPadding2D': ReflectionPadding2D,
                     'ReflectionPadding3D': ReflectionPadding3D,
                     'SFTLayer': SFTLayer,
                     'DILayer': DILayer,
                     'ms_ssim_l1_loss': ms_ssim_l1_loss}
    model = tf.keras.models.load_model(os.getcwd() + args.model, custom_objects=CUSTOM_LAYERS)
    print(model.summary())

    # sharpening
    predictor = Predictor(model, input_dir, output_dir, 224)
    predictor.make_predictions()