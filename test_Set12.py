import os.path
import logging
import os
import numpy as np
from collections import OrderedDict
from scipy.io import loadmat
from scipy import ndimage
import torch
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from models.network_ctdnet import CTDNet as net

def main():
    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    model_name = 'CTDNet_x2'
    testset_name = 'set12'
    test_sf = [2]
    show_img = False
    kernels = loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels']
    n_channels = 1
    model_pool = 'model_zoo'
    testsets = 'testsets'
    results = 'results'
    noise_level_img = 0
    noise_level_model = noise_level_img
    result_name = testset_name + '_' + model_name
    model_path = os.path.join(model_pool, model_name + '.pth')

    # ----------------------------------------
    # L_path = H_path, E_path, logger
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name)  # L_path and H_path, fixed, for Low-quality images
    E_path = os.path.join(results, result_name)  # E_path, fixed, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    model = net(n_iter=6, h_nc=32, in_nc=2, out_nc=1, nc=[16, 32, 64, 64],
                nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for key, v in model.named_parameters():
        v.requires_grad = False
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    model = model.to(device)

    logger.info('Model path: {:s}'.format(model_path))
    logger.info('Params number: {}'.format(number_parameters))
    logger.info('Model_name:{}, image sigma:{}'.format(model_name, noise_level_img))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    # --------------------------------
    # read images
    # --------------------------------
    test_results_ave = OrderedDict()
    test_results_ave['psnr_sf_k'] = []
    test_results_ave['ssim_sf_k'] = []

    for sf in test_sf:
        for k_index in range(kernels.shape[1]):
            test_results = OrderedDict()
            test_results['psnr'] = []
            test_results['ssim'] = []
            kernel = kernels[0, k_index].astype(np.float64)
            util.surf(kernel) if show_img else None
            idx = 0
            for img in L_paths:
                # --------------------------------
                # (1) classical degradation, img_L
                # --------------------------------
                idx += 1
                img_name, ext = os.path.splitext(os.path.basename(img))
                img_H = util.imread_uint(img, n_channels=n_channels)
                img_H = util.modcrop(img_H, np.lcm(sf, 8))

                # generate degraded LR image
                img_L = ndimage.filters.convolve(img_H, kernel[..., np.newaxis], mode='wrap')
                img_L = sr.downsample_np(img_L, sf, center=False)
                img_L = util.uint2single(img_L)

                np.random.seed(seed=0)
                img_L += np.random.normal(0, noise_level_img/255, img_L.shape)
                x = util.single2tensor4(img_L)
                k = util.single2tensor4(kernel[..., np.newaxis])
                sigma = torch.tensor(noise_level_model/255).float().view([1, 1, 1, 1])
                [x, k, sigma] = [el.to(device) for el in [x, k, sigma]]

                # --------------------------------
                # (2) inference
                # --------------------------------
                img_C, img_T, x = model(x, k, sf, sigma)
                # --------------------------------
                # (3) img_E
                # --------------------------------
                img_E = util.tensor2uint(x)
                img_cartoon = util.tensor2uint(img_C)
                img_texture = util.tensor2uint(img_T+0.4)
                util.imsave(img_E, os.path.join(E_path, img_name + '_x' + str(sf) + '_k' + str(
                        k_index + 1) + '_' + model_name + '.png'))
                util.imsave(img_cartoon, os.path.join(E_path, img_name + '_cartoon' + str(sf) + '_k' + str(
                    k_index + 1) + '_' + model_name + '.png'))
                util.imsave(img_texture, os.path.join(E_path, img_name + '_texture' + str(sf) + '_k' + str(
                    k_index + 1) + '_' + model_name + '.png'))
                util.imsave(img_L,
                                os.path.join(E_path, img_name + '_x' + str(sf) + '_k' + str(k_index + 1) + '_LR.png'))

                psnr = util.calculate_psnr(img_E, img_H[:, :, 0], border=sf ** 2)  # change with your own border
                ssim = util.calculate_ssim(img_E, img_H[:, :, 0], border=sf ** 2)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)
                logger.info(
                    '{:->4d}--> {:>10s} -- x{:>2d} --k{:>2d} PSNR: {:.2f}dB'.format(idx, img_name + ext, sf, k_index,
                                                                                    psnr))
                logger.info(
                    '{:->4d}--> {:>10s} -- x{:>2d} --k{:>2d} SSIM: {:.4f}'.format(idx, img_name + ext, sf, k_index,
                                                                                  ssim))
            ave_psnr_k = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim_k = sum(test_results['ssim']) / len(test_results['ssim'])
            logger.info(
                '------> Average PSNR(RGB) of ({}) scale factor: ({}), kernel: ({}) sigma: ({}): {:.2f} dB'.format(
                    testset_name, sf, k_index + 1, noise_level_model, ave_psnr_k))
            logger.info('------> Average SSIM(RGB) of ({}) scale factor: ({}), kernel: ({}) sigma: ({}): {:.4f}'.format(
                testset_name, sf, k_index + 1, noise_level_model, ave_ssim_k))
            test_results_ave['psnr_sf_k'].append(ave_psnr_k)
            test_results_ave['ssim_sf_k'].append(ave_ssim_k)
    logger.info(test_results_ave['psnr_sf_k'])
    logger.info(test_results_ave['ssim_sf_k'])


if __name__ == '__main__':
    main()
