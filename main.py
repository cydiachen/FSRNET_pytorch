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
parser.add_argument('--test_iter', type=int, default=1116, help='the interval iterations for testing')
parser.add_argument('--cdim', type=int, default=3, help='the channel-size  of the input image to network')
parser.add_argument("--nEpochs", type=int, default=1000, help="number of epochs to train for")
parser.add_argument("--start_epoch", default=161,type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--lr', type=float, default=0.7*0.7*2.5 * 10**(-4), help='learning rate, default=0.0002')
parser.add_argument('--cuda', default='True', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='./results/1_4/', help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument("--pretrained", default="/home/cydia/Music/baseline方法加入tensorboard/model/sr_1_4_0model_epoch_160_iter_0.pth", type=str, help="path to pretrained model (default: none)")

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

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


    ngpu = int(opt.ngpu)   



    #--------------build models--------------------------
    srnet = NetSR(num_layers_res=opt.num_layers_res)

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

    # calculate params
    params = list(srnet.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameter sum:" + str(k))


    criterion_l1 = nn.L1Loss(size_average=True)
    criterion_MSE = nn.MSELoss(size_average=True)
    
    if opt.cuda:
      srnet = srnet.cuda()
      criterion_l1 = criterion_l1.cuda()
      criterion_MSE = criterion_MSE.cuda()

    # optimizer_sr = optim.Adam(filter(lambda p: p.requires_grad, srnet.parameters()), lr=opt.lr, betas=(opt.momentum, 0.999), weight_decay=0.0005)
    optimizer_sr = optim.RMSprop(filter(lambda p: p.requires_grad, srnet.parameters()),lr = opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer_sr,step_size=50,gamma = 0.7)



    size = 128
    batch = 14
    save_freq = 2

    result_dir = './results/'
    result_dir0 = './results/1_4/'

    model_path = "/home/cydia/Music/baseline方法加入tensorboard/model/"

    # using tensorboardX to visualize our loss function
    writer = SummaryWriter('./log')



    input_face = np.load('/home/cydia/Documents/bisher/FSRNet/1/lr_no_noise.npy')
    train_heatmap = np.load('/home/cydia/Documents/bisher/FSRNet/1/11_parsing_maps.npy')
    train_face = np.load('/home/cydia/Documents/bisher/FSRNet/1/gts.npy')

    val_face0 = np.load('/home/cydia/Music/Baseline代码程序完善/1/val_lr1.npy')
    val_face = np.load('/home/cydia/Music/Baseline代码程序完善/1/val_gts1.npy')

    #train_heatmap = np.load('./1/parsing_maps0.npy')



    start_time = time.time()
    srnet.train()

    #----------------Train by epochs--------------------------




    for epoch in range(opt.start_epoch, opt.nEpochs):

        scheduler.step()

        loss_sr_sum = 0.0
        loss_ps_sum = 0.0
        loss_fr_sum = 0.0

        if epoch%opt.save_iter == 0:
            save_checkpoint(srnet, epoch,model_path, 0, 'sr_1_4_0')
        
        for iteration in range(len(train_face)//batch):

            target = train_face[iteration*batch:(iteration+1)*batch,:,:,:]
            input = input_face[iteration*batch:(iteration+1)*batch,:,:,:]
            target_parmap = train_heatmap[iteration*batch:(iteration+1)*batch,:,:,:]
            target = torch.from_numpy(np.float32(target)).permute(0,3,1,2)
            input = torch.from_numpy(np.float32(input)).permute(0,3,1,2)
            target_parmap = torch.from_numpy(np.float32(target_parmap)).permute(0,3,1,2)


            # --------------test-------------
            if iteration % opt.test_iter is 0 and opt.test:
                avg_psnr = 0.0
                for titer in range(len(val_face0)//batch):
                    input0 = val_face0[titer*batch:(titer+1)*batch,:,:,:]
                    # target0 = val_face[titer*batch:(titer+1)*batch,:,:,:]
                    target0 = val_face[titer*batch:(titer + 1)*batch, :, :, :]
                    target0 = torch.from_numpy(np.float32(target0)).permute(0, 3, 1, 2)
                    input0 = torch.from_numpy(np.float32(input0)).permute(0, 3, 1, 2)
                    # input, target = Variable(batch[0]), Variable(batch[1])
                    if opt.cuda:
                        input0 = input0.cuda()
                        target0 = target0.cuda()

                    with torch.no_grad():
                        output0, parsing_maps, output = forward_parallel(srnet, input0, opt.ngpu)

                    output11 = output.permute(0, 2, 3, 1).cpu().data.numpy()
                    for n in range(batch):
                        output01 = output11[n,:,:,:]
                        scipy.misc.toimage(output01 * 255, high=255, low=0, cmin=0, cmax=255).save(
                            result_dir0 + '%d_%d_%d.jpg' % (n,epoch, titer))
                        mse = criterion_MSE(output, target0)
                        psnr = 10 * log10(1 / (mse.data[0]))
                        avg_psnr += psnr

                print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / (len(val_face0))))
                writer.add_scalar('./log/Average PSNR', avg_psnr / (len(val_face0)), epoch)


            #--------------train------------
            if opt.cuda:
              input = input.cuda()
              target = target.cuda()
              target_parmap = target_parmap.cuda()

            Output0, Parsing_maps, Output = forward_parallel(srnet, input, opt.ngpu)

            loss_sr = criterion_l1(Output0, target)
            loss_ps = criterion_MSE(Parsing_maps, target_parmap)
            loss_fr = criterion_l1(Output, target)


            loss = 1000*loss_sr + 1000*loss_fr + 1000*loss_ps

            loss_sr_sum = loss_sr_sum + loss_sr.item()
            loss_ps_sum = loss_ps_sum + loss_ps.item()
            loss_fr_sum = loss_fr_sum + loss_fr.item()



            optimizer_sr.zero_grad()
            loss.backward()
            optimizer_sr.step()




            info = "===> Epoch[{}]({}/{}): time: {:4.4f}:".format(epoch, iteration, len(train_face)//batch,
                                                                  time.time() - start_time)
            info += "Rec: {:.4f}, {:.4f}, {:.4f}, Texture: {:.4f}".format(loss.data[0], loss_sr.data[0],
                                                                          loss_fr.data[0], loss_ps.data[0])

            print(info)


            if epoch % save_freq == 0:
                if not os.path.isdir(result_dir + '%04d_0' % epoch):
                    os.makedirs(result_dir + '%04d_0' % epoch)
                if not os.path.isdir(result_dir + '%04d_1' % epoch):
                    os.makedirs(result_dir + '%04d_1' % epoch)
                if not os.path.isdir(result_dir + '%04d_2' % epoch):
                    os.makedirs(result_dir + '%04d_2' % epoch)

                Output1 = Output.permute(0, 2, 3, 1).cpu().data.numpy()
                Parsing_maps0 = Parsing_maps.permute(0, 2, 3, 1).cpu().data.numpy()
                target_parmap0 = target_parmap.permute(0, 2, 3, 1).cpu().data.numpy()
                target0 = target.permute(0, 2, 3, 1).cpu().data.numpy()
                #img_predict1 = np.minimum(np.maximum(img_predict1, 0), 1)

                temp = np.concatenate((target0[0, :, :, :], Output1[0, :, :, :]), axis=1)
                temp1 = np.concatenate((target_parmap0[0, :, :, 0]/2, Parsing_maps0[0, :, :, 0]/2), axis=1)
                scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                    result_dir + '%04d_0/%d.jpg' % (epoch, iteration))
                scipy.misc.toimage(Output1[0, :, :, :] * 255, high=255, low=0, cmin=0, cmax=255).save(
                    result_dir + '%04d_2/%d.jpg' % (epoch, iteration))
                scipy.misc.toimage(temp1 * 255, high=255, low=0, cmin=0, cmax=255).save(
                    result_dir + '%04d_1/%d.jpg' % (epoch, iteration))

        # write loss into Summary Writer
        writer.add_scalar('./log/Coarse SR loss',loss_sr_sum/(len(train_face)//batch),epoch)
        writer.add_scalar('./log/Parsing Map loss',loss_ps_sum/(len(train_face)//batch),epoch)
        writer.add_scalar('./log/Fine SR loss',loss_fr_sum/(len(train_face)//batch),epoch)

def forward_parallel(net, input, ngpu):
    if ngpu > 1:
        return nn.parallel.data_parallel(net, input, range(ngpu))
    else:
        return net(input)
            
def save_checkpoint(model, epoch,model_path,iteration, prefix=""):
    model_out_path = model_path + prefix +"model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))

def save_images(images, name, path, nrow=10):   
  #print(images.size())
  img = images.cpu()
  im = img.data.numpy().astype(np.float32)
  #print(im.shape)       
  im = im.transpose(0,2,3,1)
  imsave(im, [nrow, int(math.ceil(im.shape[0]/float(nrow)))], os.path.join(path, name) )
  
def merge(images, size):
  #print(images.shape())
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))
  #print(img)
  for idx, image in enumerate(images):
    image = image * 255
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img

def imsave(images, size, path):
  img = merge(images, size)
  # print(img) 
  return cv2.imwrite(path, img)

if __name__ == "__main__":
    main()    
