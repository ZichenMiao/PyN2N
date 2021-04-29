import os
import numpy as np 
import argparse
import torch
import torch.nn as nn

from models import UNet
from dataset import load_dataset
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", default='./data/train', help='training data path')
parser.add_argument("--val_data", default='./data/val', help='validation data path')
parser.add_argument('--work_dir', default='./checkpoints', help='dir to save checkpoints and figures')
parser.add_argument('--train_size', default=-1, type=int, help='number of training images')
parser.add_argument('--val_size', default=-1, type=int, help='number of validation images')

parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epoches", type=int, default=40)
parser.add_argument("--loss_type", choices=['l1', 'l2'], default='l2')
parser.add_argument('--gpu', required=True)
parser.add_argument('--report_iter', type=int, default=100, help='iterations interval to report training stats.')

parser.add_argument('--noise_type', default='gaussian', choices=['gaussian', 'poission'])
parser.add_argument('--noise_param', type=float, default=50, help='Noise level for gaussian and poission noise')
parser.add_argument('--gaussian_mean', type=float, default=0.0, help='mean of gaussian noise')
parser.add_argument('--seed', type=int, default=3264)
parser.add_argument('--clean_tg_tr', action='store_true', help='use clean targets for training')
parser.add_argument('--crop_size', type=int, help='Random Cropped Size', default=128)

args = parser.parse_args()

## GPU constraint
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

## data
train_loader = load_dataset(args, is_train=True)
tr_iters = len(train_loader)
val_loader = load_dataset(args, is_train=False)

## model
model = UNet(in_channels=3, out_channels=3)
model.cuda()

## loss and optimizer
if args.loss_type == 'l2':
    loss = nn.MSELoss()
else:
    loss = nn.L1Loss()
loss.cuda()

optim = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 
                    patience=args.epoches//4, factor=0.5, verbose=True, mode='min')

## work_dir
work_dir = os.path.join(args.work_dir,
                    ''.join([
                    '_n2c' if args.clean_tg_tr else '_n2n',
                    '_UNet',
                    '_'+args.noise_type,
                    '_lvl{}'.format(args.noise_param),
                    '_mean{}'.format(args.gaussian_mean)
                ])
            )
os.makedirs(work_dir, exist_ok=True)
print('work dir:', work_dir)

## start training
print(args)
best_val_loss = 1000.0
best_val_psnr = 0.0
for e in range(1, args.epoches+1):
    print('\nEPOCH: {:d}/ {:d}'.format(e, args.epoches))
    
    ## train an epoch
    model.train()
    train_loss = []
    for i, (source, target, _) in enumerate(train_loader):
        source, target = source.cuda(), target.cuda()
        denoised = model(source)

        loss_ = loss(denoised, target)
        train_loss.append(loss_.detach().cpu().item())

        optim.zero_grad()
        loss_.backward()
        optim.step()

        if (i+1) % args.report_iter == 0:
            print('Iter: {}/{}, Loss: {:.4f}'.format(i+1, tr_iters, np.mean(train_loss)))
    
    ## validation
    model.eval()
    val_loss = []
    val_psnr = []
    for i, (source, target, name) in enumerate(val_loader):
        source, target = source.cuda(), target.cuda()
        denoised = model(source)

        val_loss.append(loss(denoised, target).detach().cpu().item()*source.shape[0])

        for (src, de, tg, i_nm) in zip(source, denoised, target, name):
            val_psnr.append(psnr(de, tg).detach().cpu().item())
            save_img(work_dir, de, src, tg, i_nm)


    val_loss_final = np.sum(val_loss) / args.val_size
    val_psnr_final = np.mean(val_psnr)

    print('Validation, Loss {:.4f}, PSNR: {:.2f}'.format(val_loss_final, val_psnr_final))

    ## report the best results, and save the best model
    if val_loss_final < best_val_loss:
        best_val_loss = val_loss_final
        best_val_psnr = val_psnr_final
        torch.save(model.state_dict(), os.path.join(work_dir, 'best_model.pth'))

    print('Best Validation Results: Loss {:.4f}, PSNR: {:.2f}'.format(best_val_loss, best_val_psnr))

