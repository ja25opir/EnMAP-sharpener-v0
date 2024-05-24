import argparse
import os
from model.model import Model

from config.resource_limiter import limit_tf_gpu_usage, flexible_tf_gpu_memory_growth

TILE_SIZE = 100
# NO_INPUT_BANDS = 224 + 4
# NO_OUTPUT_BANDS = 224
NO_INPUT_BANDS = 6
NO_OUTPUT_BANDS = 3
KERNEL_SIZES = [(9, 9), (3, 3), (5, 5)]

TRAIN_DATA_DIR = os.getcwd() + '/data/preprocessing/model_input/'
OUTPUT_DIR = os.getcwd() + '/output/'
LOSS_FUNCTION = 'mean_squared_error'  # todo: adapt learn rate and momentum, also use other function
BATCH_SIZE = 128  # (Masi: 128)
TRAIN_EPOCHS = 30

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start model training.')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0],
                        help='Assigned GPUs for the pipeline')
    parser.add_argument('--mem-limit', type=int, default=10, help='GPU memory limit training in GB (per GPU)')

    args = parser.parse_args()

    # todo: move to preprocess_pipeline or remove
    if os.path.exists(TRAIN_DATA_DIR + 'x/.gitkeep'):
        os.remove(TRAIN_DATA_DIR + 'x/.gitkeep')
    if os.path.exists(TRAIN_DATA_DIR + 'y/.gitkeep'):
        os.remove(TRAIN_DATA_DIR + 'y/.gitkeep')

    limit_tf_gpu_usage(args.gpus, args.mem_limit)
    # flexible_tf_gpu_memory_growth()

    cnn_model = Model(TRAIN_DATA_DIR, TILE_SIZE, NO_INPUT_BANDS, NO_OUTPUT_BANDS, BATCH_SIZE, KERNEL_SIZES,
                      LOSS_FUNCTION, TRAIN_EPOCHS, OUTPUT_DIR)

    print('Starting training...')
    cnn_model.train_model()
