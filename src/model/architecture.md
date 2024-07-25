# Architecture comparison

## training with 20 bands

### with only one skip connection:
MSE: 12.26 (predicted) vs. 15.31 (input) | 0 is perfect similarity
PSNR: 37.25 (predicted) vs. 36.28 (input) | 100 is perfect similarity
PSNR: 37.25 (predicted) vs. 36.28 (input) | 100 is perfect similarity
SSIM: 0.87 (predicted) vs. 0.83 (input) | 1.0 is perfect similarity
SSIM: 0.87 (predicted) vs. 0.83 (input) | 1.0 is perfect similarity
SAM: 1.36 (predicted) vs. 1.43 (input) | 0 is perfect similarity

### with three skip connections (ac733932e0b64ae142bdffe4511b7556a9dd5d58):
MSE: 12.15 (predicted) vs. 15.31 (input) | 0 is perfect similarity
PSNR: 37.29 (predicted) vs. 36.28 (input) | 100 is perfect similarity
PSNR: 37.29 (predicted) vs. 36.28 (input) | 100 is perfect similarity
SSIM: 0.88 (predicted) vs. 0.83 (input) | 1.0 is perfect similarity
SSIM: 0.87 (predicted) vs. 0.83 (input) | 1.0 is perfect similarity
SAM: 1.46 (predicted) vs. 1.43 (input) | 0 is perfect similarity

