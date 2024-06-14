import argparse
import os
from model.model import Model

from config.resource_limiter import limit_gpu_memory_usage, multiple_gpu_distribution

TILE_SIZE = 100
NO_INPUT_BANDS = 224 + 4
NO_OUTPUT_BANDS = 224
# NO_INPUT_BANDS = 6
# NO_OUTPUT_BANDS = 3
KERNEL_SIZES = [(9, 9), (3, 3), (5, 5)]

TRAIN_DATA_DIR = os.getcwd() + '/data/preprocessing/model_input/'
OUTPUT_DIR = os.getcwd() + '/output/'
BATCH_SIZE = 1  # (Masi: 128) # todo
LOSS_FUNCTION = 'mean_squared_error'  # todo: adapt learn rate and momentum, also use other loss function
LEARN_RATE = 0.00001 # todo: maybe use a adaptive learning rate (big steps in the beginning, small steps later)
# https://www.activeloop.ai/resources/glossary/adaptive-learning-rate-methods/#:~:text=Adaptive%20learning%20rate%20methods%20improve%20deep%20learning%20model%20performance%20by,faster%20convergence%20and%20better%20generalization.
# https://stats.stackexchange.com/questions/383807/why-we-call-adam-an-a-adaptive-learning-rate-algorithm-if-the-step-size-is-a-con
TRAIN_EPOCHS = 10

@multiple_gpu_distribution
def train_model():
    cnn_model = Model(TRAIN_DATA_DIR, TILE_SIZE, NO_INPUT_BANDS, NO_OUTPUT_BANDS, BATCH_SIZE, KERNEL_SIZES,
                      LOSS_FUNCTION, LEARN_RATE, TRAIN_EPOCHS, OUTPUT_DIR)

    print('Starting training...')
    cnn_model.train_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start model training.')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0],
                        help='Assigned GPUs for the pipeline')
    parser.add_argument('--mem-limit', type=int, default=10, help='GPU memory limit training in GB (per GPU)')

    args = parser.parse_args()

    if os.path.exists(TRAIN_DATA_DIR + 'x/.gitkeep'):
        os.remove(TRAIN_DATA_DIR + 'x/.gitkeep')
    if os.path.exists(TRAIN_DATA_DIR + 'y/.gitkeep'):
        os.remove(TRAIN_DATA_DIR + 'y/.gitkeep')

    limit_gpu_memory_usage(args.gpus, args.mem_limit)

    train_model()
