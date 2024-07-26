# Architecture comparison

## training with 20 bands

## skip connections:

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

### skip connections before feature injection, normal detail branch (4f420097ba6fd779b86eb18cc1bead8439b5ac39):

loss: 11193, acc: 0.57
MSE: 12.68 (predicted) vs. 15.31 (input) | 0 is perfect similarity
PSNR: 37.10 (predicted) vs. 36.28 (input) | 100 is perfect similarity
SSIM: 0.87 (predicted) vs. 0.83 (input) | 1.0 is perfect similarity
SAM: 1.66 (predicted) vs. 1.43 (input) | 0 is perfect similarity
-> a little worse
-> skip connections between the layers don't seem to have a big impact

## detail branch:

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

best atm:

- 3, 3, 3 feature maps in detail branch
- (3, 3) kernels in detail branch
- 1 skip connection (at the end)

## loss function:

### MS SSIM L1 loss:

https://arxiv.org/pdf/1511.08861
loss: 8.45, acc: 0.64
MSE: 12.41 (predicted) vs. 15.31 (input) | 0 is perfect similarity
PSNR: 37.19 (predicted) vs. 36.28 (input) | 100 is perfect similarity
SSIM: 0.87 (predicted) vs. 0.83 (input) | 1.0 is perfect similarity
SAM: 1.30 (predicted) vs. 1.43 (input) | 0 is perfect similarity
--> better SAM than with MSE loss

## training with 60 bands (20:80)

### detail branch (3,3,3), main branch (64,32,9,1), 3 skip connections, MS SSIM L1 loss:

loss: 21.77, acc: 0.67
MSE: 34.02 (predicted) vs. 40.49 (input) | 0 is perfect similarity
PSNR: 32.81 (predicted) vs. 32.06 (input) | 100 is perfect similarity
SSIM: 0.82 (predicted) vs. 0.76 (input) | 1.0 is perfect similarity
SAM: 4.02 (predicted) vs. 4.33 (input) | 0 is perfect similarity

### WITHOUT usage of the detail branch:

loss: 21.36, acc: 0.68
MSE: 34.01 (predicted) vs. 40.49 (input) | 0 is perfect similarity
PSNR: 32.81 (predicted) vs. 32.06 (input) | 100 is perfect similarity
SSIM: 0.82 (predicted) vs. 0.76 (input) | 1.0 is perfect similarity
SAM: 3.98 (predicted) vs. 4.33 (input) | 0 is perfect similarity
--> slightly better than with detail branch

### detail branch with ADD instead of STACK:

loss:



## training with 20 bands (20:40)

loss:

## training with 20 bands (200:220)

loss:

## residual learning see: https://openaccess.thecvf.com/content_cvpr_2016/papers/Kim_Accurate_Image_Super-Resolution_CVPR_2016_paper.pdf
