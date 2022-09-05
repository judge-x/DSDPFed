import torch
import numpy as np

def GaussianMechanism(grad_tensor, sigma_g, use_cuda):
    """ Adds Gaussian noise to per-sample stochastic gradients.
    :param grad_tensor : stands for a stochastic gradient
    :param sigma_g : variance of the Gaussian noise (defined by DP theory)
    :param max_norm : clipping value
    :param batch_size : nb of data point in the considered batch"""
    std_gaussian = sigma_g
    gaussian_noise = torch.normal(0, std_gaussian, grad_tensor.shape)

    if use_cuda:
        gaussian_noise = gaussian_noise.cuda()

    return grad_tensor + gaussian_noise
