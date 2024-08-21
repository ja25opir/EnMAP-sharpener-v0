import argparse, os
import tensorflow as tf
from model.architecture import ReflectionPadding2D, ReflectionPadding3D, SFTLayer, DILayer, ms_ssim_l1_loss
from model.test_model import Evaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start EnMAP sharpening.')
    parser.add_argument('--model', type=str, default='/output/models/supErMAPnet.keras')
    parser.add_argument('--validation-data-dir', type=str, default='/data/validation/',
                        help='Path to the directory with input and corresponding ground truth data')

    args = parser.parse_args()

    validation_data_dir = os.getcwd() + args.validation_data_dir

    # load model with its custom layers
    CUSTOM_LAYERS = {'ReflectionPadding2D': ReflectionPadding2D,
                     'ReflectionPadding3D': ReflectionPadding3D,
                     'SFTLayer': SFTLayer,
                     'DILayer': DILayer,
                     'ms_ssim_l1_loss': ms_ssim_l1_loss}
    model = tf.keras.models.load_model(os.getcwd() + args.model, custom_objects=CUSTOM_LAYERS)
    print(model.summary())

    # evaluation
    evaluator = Evaluator(model, validation_data_dir, 224)
    # evaluator.evaluate_model()
    evaluator.plot_model_graph(output_dir=os.getcwd() + '/output/figures/')