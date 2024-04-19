from model.model import Model
import os

TILE_SIZE = 100
NO_INPUT_BANDS = 224 + 4
NO_OUTPUT_BANDS = 224
KERNEL_SIZES = [(9, 9), (5, 5), (5, 5)]

TRAIN_DATA_DIR = os.getcwd() + '/data/preprocessing/model_input/'
LOSS_FUNCTION = 'mean_squared_error'  # todo
BATCH_SIZE = 32  # (Masi: 128)

if __name__ == '__main__':
    # todo: move to preprocess_pipeline
    if os.path.exists(TRAIN_DATA_DIR + 'x/.gitkeep'):
        os.remove(TRAIN_DATA_DIR + 'x/.gitkeep')
    if os.path.exists(TRAIN_DATA_DIR + 'y/.gitkeep'):
        os.remove(TRAIN_DATA_DIR + 'y/.gitkeep')


    cnn_model = Model(TRAIN_DATA_DIR, TILE_SIZE, NO_INPUT_BANDS, NO_OUTPUT_BANDS, BATCH_SIZE, KERNEL_SIZES,
                      LOSS_FUNCTION)

    print('Starting training...')
    cnn_model.train_model()
