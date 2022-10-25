# [CTDNet: Cartoon-texture decomposition-based gray image super-resolution network with multiple degradations]


# Requirements and Dependencies
- Pytorch 1.7.1
- MATLAB R2018a

# [Testing code](https://github.com/shibaoshun/CTDNet) 

`We provide the test codes on Set12 and real Terahertz images. We also upload the BAR method to post-process the SR Terahertz images.`

- [test_Set12.py](https://github.com/shibaoshun/CTDNet/test_Set12.py)
  
- [test_THZ.py](https://github.com/shibaoshun/CTDNet/test_THZ.py)

- [BAR.m](https://github.com/shibaoshun/CTDNet/BAR/BAR.m)


# Testing (Python and Matlab)

- [demos]  `test_Set12.py`, `test_THZ.py`, `BAR.m`.

- [models]  including a trained model for x2 SR in model_zoo.

- [testsets]  Set12 and real Terahertz images.
   

# Results

## Comparisons with existing methods

**Average PSNR(dB)/SSIM values of different SR algorithms with noise level 0, 5 and 10 on Set12. The reported results are the average PSNR(dB)/SSIM values for scale factors of 2, 3 and 4. The average means to calculate the average values of each algorithm for different scale factors.**

| noise level 0 | 2 | 3  | 4 | Average |  
|:-------:|:-------:|:-------:|:-------:|:-------:|
| VDSR  |  26.26/0.7841  |   25.40/0.7556   | 24.38/0.7120  |   25.35/0.7506   |
| PAN  |  27.74/0.8369  |   25.86/0.7854   | 24.72/0.7331  | 26.11/0.7851 |   
| RCAN  |  30.00/0.8926  |   26.72/0.8145   | 25.27/0.7596  | 27.33/0.8222 |  
| SRMD  |  30.11/0.8888  |   27.05/0.8197   | 25.55/0.7652  |   27.57/0.8246   | 
| USRNet  |  31.38/0.9036  |   28.04/0.8425   | 25.56/0.7649  | 28.33/0.8370 |   
| CarNet_only  |  28.77/0.8360  |   26.20/0.7729   | 24.42/0.7099  | 26.46/0.7729 |   
| TexNet_only  |  31.79/0.9098  |   28.05/0.8418   | 25.99/0.7852  | 28.61/0.8456 |   
| CTDnet  |  32.06/0.9163  |   28.34/0.8511   | 26.07/0.7868  | **28.82**/**0.8514** | 

| noise level 5 | 2 | 3  | 4 | Average |  
|:-------:|:-------:|:-------:|:-------:|:-------:|
| VDSR  |  26.13/0.7749  |   25.30/0.7475   | 24.30/0.7048  |   25.24/0.7424  |
| PAN  |  26.99/0.7978  |   25.45/0.7575   | 24.41/0.7130  | 25.62/0.7561 |   
| RCAN  |  28.03/0.8286  |   26.06/0.7769   | 24.77/0.7300  | 26.29/0.7785 |  
| SRMD  |  28.33/0.8232  |   26.39/0.7792   | 25.07/0.7347  |  26.60/0.7790 | 
| USRNet  |  28.69/0.8292  |   26.99/0.7961   | 25.11/0.7367  | 26.93/0.7873 |   
| CarNet_only  |  27.51/0.7943  |   25.75/0.7523   | 24.22/0.6990  | 25.83/0.7485 |   
| TexNet_only  |  28.92/0.8375  |   27.03/0.7964   | 25.37/0.7515  | 27.11/0.7951 |   
| CTDnet  |  29.01/0.8401  |   27.12/0.8001   | 25.45/0.7513  | **27.19**/**0.7972** | 

| noise level 10 | 2 | 3  | 4 | Average |  
|:-------:|:-------:|:-------:|:-------:|:-------:|
| VDSR  |  25.83/0.7546  |   25.04/0.7286   | 24.12/0.6878  |   25.00/0.7237  |
| PAN  |  26.09/0.7625  |   24.77/0.7249   | 23.82/0.6848  | 24.89/0.7241 |   
| RCAN  |  26.89/0.7900  |   25.22/0.7400   | 23.98/0.6956  | 25.33/0.7419 |  
| SRMD  |  27.07/0.7838  |   25.46/0.7415   | 24.38/0.7036  |  25.64/0.7430 | 
| USRNet  |  27.24/0.7866  |   25.88/0.7559   | 24.39/0.7051  | 25.84/0.7492 |   
| CarNet_only  |  26.21/0.7450  |   24.80/0.7055   | 23.57/0.6638  | 24.86/0.7048 |   
| TexNet_only  |  27.42/0.7964  |   25.93/0.7567   | 24.58/0.7166  | 25.98/0.7566 |   
| CTDnet  |  27.49/0.7983  |   25.95/0.7593   | 24.62/0.7163  | **26.02**/**0.7580** | 


## Visual Results

For visual results, please see the supplementary material: [Supplementary Material.pdf](https://github.com/shibaoshun/CTDNet/Supplementary_Material.pdf)





