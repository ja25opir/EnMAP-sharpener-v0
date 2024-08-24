import os, time
import numpy as np

def make_prediction(x, x1, model, output_bands):
    x = x.T.reshape(1, 32, 32, 4)
    x1 = x1.T.reshape(1, 32, 32, output_bands, 1)
    return model.predict({'x': x, 'x1': x1}, verbose=0).reshape(32, 32, output_bands).T

class Predictor:
    def __init__(self, model, input_data_path, output_data_path, no_output_bands):
        self.model = model
        self.x_data_path = input_data_path
        self.output_data_path = output_data_path
        self.no_output_bands = no_output_bands

    def load_data(self, file_name):
        x = np.load(self.x_data_path + file_name)
        x = x[(224, 225, 226, 227), :, :]
        x1 = np.load(self.x_data_path + file_name)[:224, :, :]
        # x1 = np.load(X_DATA_PATH + file_name)[20:60, :, :]  # 40 bands only
        # y = y[20:60, :, :]  # 40 bands only
        return x, x1

    def make_predictions(self):
        """Make predictions on the input data"""
        i = 1
        files = os.listdir(self.x_data_path)
        no_files = len(files)
        print(f'Starting predictions for {no_files} files...')
        start = time.time()
        for file_name in files:
            x_rast, x1_rast = self.load_data(file_name)
            predicted_rast = make_prediction(x_rast, x1_rast, self.model, self.no_output_bands)
            np.save(self.output_data_path + file_name, predicted_rast)
            if i % 25 == 0:
                print(f'Predicted {i} / {no_files} files')
            i += 1
        print(f'Predicted {no_files} files in {time.time() - start} seconds')
        print(f'Average prediction time per file: {(time.time() - start) / no_files} seconds')
