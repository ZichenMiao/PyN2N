import torch
import numpy as np 
import os
from PIL import Image
import torchvision.transforms as T

def save_model(params, model):
    pass


def psnr(denoised, target):
    ## source, target: a pair of images with range [0.0, 1.0]
    return 10 * torch.log10(1. / torch.mean((denoised-target)**2))

def save_img(work_dir, denoised, noised, clean, img_name):
    # import pdb; pdb.set_trace()
    save_dir = os.path.join(work_dir, 'val_visualization')
    os.makedirs(save_dir, exist_ok=True)
    denoised, noised, clean = T.functional.to_pil_image(denoised), T.functional.to_pil_image(noised), T.functional.to_pil_image(clean)
    pre, suff = img_name.split('.')
    denoised.save(os.path.join(save_dir, pre+'_denoise.'+suff))
    noised.save(os.path.join(save_dir, pre+'_noised.'+suff))
    clean.save(os.path.join(save_dir, pre+'_clean.'+suff))

