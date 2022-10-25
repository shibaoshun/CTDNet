import os.path
import logging
import numpy as np
from scipy.io import loadmat
import scipy.signal as correlate
import torch
from utils import utils_logger
from utils import utils_image as util
from utils import utils_model

def ch4_ch2(x):##通道
    x_real = x[0, :, :]
    x_imag = x[1, :, :]
    x = torch.complex(x_real, x_imag)
    return x
def function_stdEst2D(y):
    # [d, m, n] = z.shape
    # if d == 2:
    #     y = ch4_ch2(z[0, :, :, :])
    # else:
    #     y1 = torch.squeeze(z, 0)
    #     y = torch.squeeze(y1, 0)
    # y = z
    # y = y.detach().numpy()
    daub6kern = np.array([0.03522629188571, 0.08544127388203, -0.13501102001025, -0.45987750211849, 0.80689150931109, -0.33267055295008]).reshape(1, 6)
    wav_det = correlate.correlate2d(y, daub6kern, 'same')
    wav_det = correlate.correlate2d(wav_det, daub6kern.T)
    dev = np.median(abs(wav_det[:])) / .6745
    return dev

def main():
    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    model_name = 'CTDNet_x2'
    testset_name = 'THZ'
    sf = 2
    x8 = False                           # default: False, x8 to boost performance
    n_channels = 1            # fixed
    model_pool = 'model_zoo'  # fixed
    testsets = 'testsets'     # fixed
    results = 'results'       # fixed
    result_name = testset_name + '_' + model_name
    model_path = os.path.join(model_pool, model_name+'.pth')
    kernels = loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels']

    # ----------------------------------------
    # L_path, E_path
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    from models.network_ctdnet import CTDNet as net
    model = net(n_iter=6, h_nc=32, in_nc=2, out_nc=1, nc=[16, 32, 64, 64],
                nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    for k_index in range(kernels.shape[1]):
        kernel = kernels[0, k_index].astype(np.float64)
        for idx, img in enumerate(L_paths):
            # ------------------------------------
            # (1) img_L
            # ------------------------------------
            img_name, ext = os.path.splitext(os.path.basename(img))
            img_L = util.imread_uint(img, n_channels=n_channels)
            img_L = util.uint2single(img_L)
            noise_level_model=8*function_stdEst2D(img_L[:,:,0])
            print(noise_level_model)
            x = util.single2tensor4(img_L)
            k = util.single2tensor4(kernel[..., np.newaxis])
            sigma = torch.tensor(noise_level_model).float().view([1, 1, 1, 1])
            [x, k, sigma] = [el.to(device) for el in [x, k, sigma]]

            # ------------------------------------
            # (2) img_E
            # ------------------------------------
            if not x8:
                img_C, img_T, img_E = model(x, k, sf, sigma)
            else:
                img_C, img_T, img_E = utils_model.test_mode(model, img_L, mode=3, sf=sf)
            img_C = util.tensor2uint(img_C)
            img_T = util.tensor2uint(img_T + 0.4)
            img_E = util.tensor2uint(img_E)

            # ------------------------------------
            # (3) save results
            # ------------------------------------
            util.imsave(img_C, os.path.join(E_path, img_name + '_cartoon' + str(sf) + '_k' + str(
                k_index + 1) + '_' + model_name + '.png'))
            util.imsave(img_T, os.path.join(E_path, img_name + '_texture' + str(sf) + '_k' + str(
                k_index + 1) + '_' + model_name + '.png'))
            util.imsave(img_E, os.path.join(E_path, img_name + '_x' + str(sf) + '_k' + str(
                k_index + 1) + '_' + model_name + '.png'))

if __name__ == '__main__':

    main()

