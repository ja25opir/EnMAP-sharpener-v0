import argparse
import os
import time
from model.model import Model

from config.resource_limiter import limit_gpu_memory_usage, multiple_gpu_distribution

TILE_SIZE = 32
# TILE_SIZE = 10
# NO_INPUT_BANDS = 224 + 4
# NO_OUTPUT_BANDS = 224
NO_INPUT_BANDS = 4
NO_OUTPUT_BANDS = 40

OUTPUT_DIR = os.getcwd() + '/output/'
LOSS_FUNCTION = 'mean_squared_error'


# LEARN_RATE = 0.00001
# https://www.activeloop.ai/resources/glossary/adaptive-learning-rate-methods/#:~:text=Adaptive%20learning%20rate%20methods%20improve%20deep%20learning%20model%20performance%20by,faster%20convergence%20and%20better%20generalization.
# https://stats.stackexchange.com/questions/383807/why-we-call-adam-an-a-adaptive-learning-rate-algorithm-if-the-step-size-is-a-con

# @multiple_gpu_distribution
@multiple_gpu_distribution
def train_model(batch_size, epochs, train_data):
    cnn_model = Model(train_data, TILE_SIZE, NO_INPUT_BANDS, NO_OUTPUT_BANDS, batch_size,
                      LOSS_FUNCTION, epochs, OUTPUT_DIR)

    print('Starting training...')
    # cnn_model.train_model()
    cnn_model.fit_hyperparameter()
    return cnn_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start model training.')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0],
                        help='Assigned GPUs for the pipeline')
    parser.add_argument('--mem-limit', type=int, default=10, help='GPU memory limit training in GB (per GPU)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--train-data-dir', type=str, default='/data/preprocessing/model_input/',
                        help='Path to the directory with training data')

    args = parser.parse_args()

    train_data_dir = os.getcwd() + args.train_data_dir
    if os.path.exists(train_data_dir + 'x/.gitkeep'):
        os.remove(train_data_dir + 'x/.gitkeep')
    if os.path.exists(train_data_dir + 'y/.gitkeep'):
        os.remove(train_data_dir + 'y/.gitkeep')

    limit_gpu_memory_usage(args.gpus, args.mem_limit)

    start = time.time()
    model = train_model(args.batch_size, args.epochs, train_data_dir)
    end = time.time()

    model.plot_history()
    print("---ModelTraining---Elapsed time: %.2fs seconds ---" % (end - start))
