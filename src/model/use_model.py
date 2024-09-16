import os, time
import pickle
import numpy as np
import rasterio


def prediction(x, x1, model, output_bands):
    """Make prediction for a single input tile."""
    x = x.T.reshape(1, 32, 32, 4)
    x1 = x1.T.reshape(1, 32, 32, output_bands, 1)
    start = time.time()
    pred = model.predict({'x': x, 'x1': x1}, verbose=0).reshape(32, 32, output_bands).T
    prediction_time = time.time() - start
    return pred, prediction_time


class Predictor:
    """
    Class for making predictions for all files in a given directory.
    """
    def __init__(self, model, input_data_path, output_data_path, no_output_bands):
        self.model = model
        self.x_data_path = input_data_path
        self.output_data_path = output_data_path
        self.no_output_bands = no_output_bands

    def load_data(self, file_name):
        """Load single tile, return HSI and MSI data for model branches."""
        x = np.load(self.x_data_path + file_name)
        x = x[(224, 225, 226, 227), :, :]
        x1 = np.load(self.x_data_path + file_name)[:224, :, :]
        # x1 = np.load(self.x_data_path + file_name)[20:60, :, :]  # 40 bands only
        return x, x1

    def make_predictions(self):
        """Make predictions for all files in a given directory."""
        i = 1
        files = os.listdir(self.x_data_path)
        # remove pickled meta files from list
        files = [f for f in files if f.endswith('.npy')]
        no_files = len(files)
        print(f'Starting predictions for {no_files} files...')
        prediction_times = []
        start = time.time()
        for file_name in files:
            x_rast, x1_rast = self.load_data(file_name)
            predicted_rast, prediction_time = prediction(x_rast, x1_rast, self.model, self.no_output_bands)
            prediction_times.append(prediction_time)
            np.save(self.output_data_path + file_name, predicted_rast)
            if i % 25 == 0:
                print(f'Predicted {i} / {no_files} files')
            i += 1

        print(f'Average prediction time: {np.mean(prediction_times):.5f} seconds')
        print(f'Predicted + saved {no_files} files in {time.time() - start} seconds')
        print(f'Average prediction + saving time per file: {(time.time() - start) / no_files} seconds')


class Reconstructor:
    """
    Class for reconstructing full scenes from tiles.
    """
    def __init__(self, predictions_path, meta_path, output_path):
        self.predictions_path = predictions_path
        self.meta_path = meta_path
        self.output_path = output_path
        self.tile_size = 32
        self.tiling_margin = 5

    def reconstruct_all(self):
        """Reconstruct all scenes in a given directory."""
        all_files = os.listdir(self.predictions_path)
        all_tiles = [f for f in all_files if f.endswith('.npy')]
        # get unique scene timestamps
        scenes = list(set([f.split('_')[0] for f in all_tiles]))
        # reconstruct each scene
        for scene in scenes:
            print(f'Reconstructing scene {scene}...')
            scene_tiles = [f for f in all_tiles if f.startswith(scene)]
            start = time.time()
            self.reconstruct_image(scene_tiles, scene)
            print(f'Reconstructed scene {scene} in {time.time() - start} seconds')

    def reconstruct_image(self, predictions_list, scene_timestamp):
        """Reconstruct a full scene from its tiles."""
        # sort by tile numbers
        predictions_list.sort(key=lambda p: int(p.split('_')[2].strip('.npy')))
        predictions_list.sort(key=lambda p: int(p.split('_')[1]))

        # read original meta from file
        with open(self.meta_path + scene_timestamp + '_meta.pkl', 'rb') as f:
            meta = pickle.load(f)

        max_x = (meta['width'] - self.tiling_margin) // self.tile_size
        max_y = (meta['height'] - self.tiling_margin) // self.tile_size

        # create empty matrix
        reconstruction_mat = np.zeros((224, max_y * self.tile_size, max_x * self.tile_size))

        # update meta to match number of output bands and output resolution
        meta.update(count=224)
        meta.update(width=max_x * self.tile_size)
        meta.update(height=max_y * self.tile_size)
        meta.update(transform=rasterio.Affine(10.0, 0.0, meta['transform'][2],
                                              0.0, -10.0, meta['transform'][5]))

        # use max_y for x as rasterio swaps axis
        for x in range(0, max_y):
            for y in range(0, max_x):
                tile_name = f'{scene_timestamp}_{y}_{x}.npy'
                try:
                    tile = np.load(self.predictions_path + tile_name)
                except FileNotFoundError:
                    print(f'{tile_name} not found')
                    continue
                reconstruction_mat[:, x * self.tile_size:(x + 1) * self.tile_size,
                y * self.tile_size:(y + 1) * self.tile_size] = tile
            print(f'row {x + 1} of {max_y} done')

        # save matrix as raster
        with rasterio.open(self.output_path + scene_timestamp + '.tif', "w",
                           **meta) as dest:
            dest.write(reconstruction_mat)
