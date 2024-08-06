import os
import numpy as np
import matplotlib.pyplot as plt

"""plot hyperparam search results"""


def plot_model_histories(hist, hyperparams='kernels'):
    keys = []
    if hyperparams == 'kernels':
        keys = ["k_mb", "k_db"]
    if hyperparams == 'filters':
        keys = ["f_mb", "f_db"]

    for model in hist:
        plt.plot(model['hist']['val_ssim'], label=f'{model[keys[0]] + model[keys[1]]}')
    # plt.legend()
    plt.show()

    for model in hist:
        plt.plot(model['hist']['val_psnr'], label=f'{model[keys[0]] + model[keys[1]]}')
    plt.show()


def print_best_metrics(hist, hyperparams="kernels"):
    keys = []
    if hyperparams == 'kernels':
        keys = ["k_mb", "k_db"]
    if hyperparams == 'filters':
        keys = ["f_mb", "f_db"]

    metrics = []
    for model in hist:
        metrics.append({hyperparams: model[keys[0]] + model[keys[1]],
                        'ssim': model['hist']['ssim'][-1],
                        'psnr': model['hist']['psnr'][-1],
                        'mse': model['hist']['mse'][-1],
                        'loss': model['hist']['loss'][-1]})

    best_ssim = max(metrics, key=lambda x: x['ssim'])
    best_psnr = max(metrics, key=lambda x: x['psnr'])
    best_mse = min(metrics, key=lambda x: x['mse'])
    best_loss = min(metrics, key=lambda x: x['loss'])

    print(f'Best SSIM: {best_ssim}, {hyperparams}: {best_ssim[hyperparams]}')
    print(f'Best PSNR: {best_psnr}, {hyperparams}: {best_psnr[hyperparams]}')
    print(f'Best MSE: {best_mse}, {hyperparams}: {best_mse[hyperparams]}')
    print(f'Best Loss: {best_loss}, {hyperparams}: {best_loss[hyperparams]}')
    print('----------')

    val_metrics = []
    for model in hist:
        val_metrics.append({hyperparams: model[keys[0]] + model[keys[1]],
                            'val_ssim': model['hist']['val_ssim'][-1],
                            'val_psnr': model['hist']['val_psnr'][-1],
                            'val_mse': np.round(model['hist']['val_mse'][-1], 2),
                            'val_loss': np.round(model['hist']['val_loss'][-1], 5)})

    print('Validation metrics:')
    val_metrics = sorted(val_metrics, key=lambda x: x['val_ssim'], reverse=True)
    print(f'Best SSIM: {val_metrics[0]["val_ssim"]}, {hyperparams}: {val_metrics[0][hyperparams]}')
    print(f'Second best SSIM: {val_metrics[1]["val_ssim"]}, {hyperparams}: {val_metrics[1][hyperparams]}')
    val_metrics = sorted(val_metrics, key=lambda x: x['val_psnr'], reverse=True)
    print(f'Best PSNR: {val_metrics[0]["val_psnr"]}, {hyperparams}: {val_metrics[0][hyperparams]}')
    print(f'Second best PSNR: {val_metrics[1]["val_psnr"]}, {hyperparams}: {val_metrics[1][hyperparams]}')
    val_metrics = sorted(val_metrics, key=lambda x: x['val_mse'])
    print(f'Best MSE: {val_metrics[0]["val_mse"]}, {hyperparams}: {val_metrics[0][hyperparams]}')
    print(f'Second best MSE: {val_metrics[1]["val_mse"]}, {hyperparams}: {val_metrics[1][hyperparams]}')
    val_metrics = sorted(val_metrics, key=lambda x: x['val_loss'])
    print(f'Best Loss: {val_metrics[0]["val_loss"]}, {hyperparams}: {val_metrics[0][hyperparams]}')
    print(f'Second best Loss: {val_metrics[1]["val_loss"]}, {hyperparams}: {val_metrics[1][hyperparams]}')


# kernels_history_path = os.getcwd() + '/../../../output/models/MMSRes_hyperparam_history_kernels.txt'
# with open(kernels_history_path, 'r') as f:
#     # read dict from file
#     kernels_history = eval(f.read())
# plot_model_histories(kernels_history, hyperparams='kernels')
# print_best_metrics(kernels_history, hyperparams='kernels')

filters_history_path = os.getcwd() + '/../../../output/models/MMSRes_hyperparam_history_filters.txt'
with open(filters_history_path, 'r') as f:
    # read dict from file
    filters_history = eval(f.read())
plot_model_histories(filters_history, hyperparams='filters')
print_best_metrics(filters_history, hyperparams='filters')
