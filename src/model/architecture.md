# Architecture comparison

## training with 20 bands

### with only one skip connection:

MSE: 12.26 (predicted) vs. 15.31 (input) | 0 is perfect similarity
PSNR: 37.25 (predicted) vs. 36.28 (input) | 100 is perfect similarity
SSIM: 0.87 (predicted) vs. 0.83 (input) | 1.0 is perfect similarity
SAM: 1.36 (predicted) vs. 1.43 (input) | 0 is perfect similarity

### with three skip connections (ac733932e0b64ae142bdffe4511b7556a9dd5d58):

loss: 11158, acc: 0.61
MSE: 12.15 (predicted) vs. 15.31 (input) | 0 is perfect similarity
PSNR: 37.29 (predicted) vs. 36.28 (input) | 100 is perfect similarity
SSIM: 0.87 (predicted) vs. 0.83 (input) | 1.0 is perfect similarity
SAM: 1.46 (predicted) vs. 1.43 (input) | 0 is perfect similarity

### 9, 6, 3 feature maps in detail branch:
loss: 12181, acc: 0.58
MSE: 12.28 (predicted) vs. 15.31 (input) | 0 is perfect similarity
PSNR: 37.24 (predicted) vs. 36.28 (input) | 100 is perfect similarity
SSIM: 0.87 (predicted) vs. 0.83 (input) | 1.0 is perfect similarity
SAM: 1.55 (predicted) vs. 1.43 (input) | 0 is perfect similarity
--> worse

### detail branch kernels (9,9) (6,6) (3,3):
loss: 13555, acc: 0.59
MSE: 12.83 (predicted) vs. 15.31 (input) | 0 is perfect similarity
PSNR: 37.05 (predicted) vs. 36.28 (input) | 100 is perfect similarity
SSIM: 0.87 (predicted) vs. 0.83 (input) | 1.0 is perfect similarity
SAM: 1.70 (predicted) vs. 1.43 (input) | 0 is perfect similarity
--> worse

### skip connections before feature injection, normal detail branch:
loss: