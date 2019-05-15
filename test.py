from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from dataset import *
import time
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
from networks import *
from math import log10
import torchvision
import cv2
import skimage
import scipy.io
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--test', default='True', action='store_true', help='enables test during training')
parser.add_argument('--mse_avg', action='store_true', help='enables mse avg')
parser.add_argument('--num_layers_res', type=int, help='number of the layers in residual block', default=2)
parser.add_argument('--nrow', type=int, help='number of the rows to save images', default=1)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--test_batchSize', type=int, default=64, help='test batch size')
parser.add_argument('--save_iter', type=int, default=10, help='the interval iterations for saving models')
parser.add_argument('--test_iter', type=int, default=500, help='the interval iterations for testing')
parser.add_argument('--cdim', type=int, default=3, help='the channel-size  of the input image to network')
parser.add_argument("--nEpochs", type=int, default=1000, help="number of epochs to train for")
parser.add_argument("--start_epoch", default=0, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--lr', type=float, default=0.7 * 2.5 * 10 ** (-4), help='learning rate, default=0.0002')
parser.add_argument('--cuda', default='True', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='./results/1_4/', help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument("--pretrained",default="/home/cydia/Music/baseline方法加入tensorboard/model/sr_1_4_0model_epoch_150_iter_0.pth", type=str,
                    help="path to pretrained model (default: none)")


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    criterion_l1 = nn.L1Loss(size_average=True)
    criterion_MSE = nn.MSELoss(size_average=True)


    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ngpu = int(opt.ngpu)

    # --------------build models--------------------------
    with torch.no_grad():
        srnet = NetSR(num_layers_res=opt.num_layers_res)

    if opt.cuda:
        srnet = srnet.cuda()
        criterion_l1 = criterion_l1.cuda()
        criterion_MSE = criterion_MSE.cuda()


    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)

            # debug
            print("现在我们对于与训练模型进行debug")
            print(weights)
            pretrained_dict = weights['model'].state_dict()

            model_dict = srnet.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)

            # 3. load the new state dict
            srnet.load_state_dict(model_dict)
            # srnet.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("Display Network Strcuture:")
    print(srnet)

    # # calculate params
    # params = list(srnet.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     for j in i.size():
    #         l *= j
    #     k = k + l
    # print("total parameter sum:" + str(k))
    size = 128
    batch = 14
    save_freq = 2

    result_dir = './results/'
    result_dir0 = './results/1_4/'

    model_path = "/home/cydia/Music/baseline测试部分有错，修改/model/"

    # using tensorboardX to visualize our loss function
    writer = SummaryWriter('./log')
    # val_face = np.load('/home/cydia/Documents/bisher/FSRNet/1/gts1.npy')
    val_face0 = np.load('/home/cydia/Documents/bisher/FSRNet/1/lr1.npy')

    # srnet.train()
    avg_psnr = 0.0

    LENGTH = val_face0.shape[0] // batch

    # for titer in range(LENGTH):
    #
    #     # for titer in range(0,LENGTH):

    for titer in range(LENGTH):

        input0 = val_face0[titer * batch:(titer + 1) * batch, :, :, :]
        input0 = torch.from_numpy(np.float32(input0)).permute(0, 3, 1, 2)

        if opt.cuda:
            input0 = input0.cuda()
        try:
            with torch.no_grad():
                output0, parsing_maps, output = srnet(input0)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("Warning: out of memory")
                if hasattr(torch.cuda,'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception


        output11 = output.permute(0, 2, 3, 1).cpu().data.numpy()

        for n in range(batch):
            output01 = output11[n, :, :, :]
            scipy.misc.toimage(output01 * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir0 + 'lr_%d_%d.jpg' % (titer, n))


if __name__ == "__main__":
    main()
