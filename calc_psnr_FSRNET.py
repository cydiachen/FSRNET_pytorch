import cv2
import os
import numpy as np
import math
from PIL import Image
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import pandas as pd
import skimage


csv_dir = "/home/cydia/Music/baseline方法加入tensorboard/fsr_eval_celeba140.csv"
fsr_dir = "/home/cydia/Music/baseline方法加入tensorboard/results/0140_0/"



def create_csv():
    df = pd.DataFrame(columns = ["filename","psnr","mse","ssim"])
    df.to_csv(csv_dir,index = False)
    print (df)

#导入你要测试的图像
# input: full image name
# output: tuple ,using (img1,img2) to receive
def cv2_separate_images(image_name):
    img = cv2.imread(image_name)
    img1 = np.zeros(((128, 128, 3)))
    img2 = np.zeros(((128, 128, 3)))
    for i in range(0, 128):
        for j in range(0, 128):
            img1[i, j, :] = img[i, j, :]
    for i in range(0, 128):
        for j in range(128, 256):
            img2[i, j - 128, :] = img[i, j, :]
    return img1,img2

# function to process
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
def calc_PSNR(output,gt):
    im1 = np.array(output,'f')
    im2 = np.array(gt,'f')

    #图像的行数
    height = im1.shape[0]
    #图像的列数
    width = im1.shape[1]


    #提取R通道
    r = im1[:,:,0]
    #提取g通道
    g = im1[:,:,1]
    #提取b通道
    b = im1[:,:,2]
    #打印g通道数组
    #print (g)
    #图像1,2各自分量相减，然后做平方；
    R = im1[:,:,0]-im2[:,:,0]
    G = im1[:,:,1]-im2[:,:,1]
    B = im1[:,:,2]-im2[:,:,2]
    #做平方
    mser = R*R
    mseg = G*G
    mseb = B*B
    #三个分量差的平方求和
    SUM = mser.sum() + mseg.sum() + mseb.sum()
    MSE = SUM / (height * width * 3)
    PSNR = 10*math.log ( (255.0*255.0/(MSE)) ,10)

    #print (PSNR)
    return PSNR

# 保存benchmark结果
def save_result(csv_name,filename,psnr,mse,ssim):
    #i = 0
    df = pd.read_csv(csv_name)
    data = {"filename":filename,"psnr":psnr,"mse":mse,"ssim":ssim}
    df = df.append(data,ignore_index = True)
    df.to_csv(csv_name,index = False)


def main():

    # Go through all the pics in folders

    # 建立csv database
    if not os.path.exists(csv_dir):
        create_csv()

    # 遍历fsr目录，读取文件名，保存到filename
    for dirpath,dirnames,filenames in os.walk(fsr_dir):
        filenames.sort()
        for file in filenames:
            fullpath = os.path.join(dirpath,file)
            (img1,img2) = cv2_separate_images(fullpath)

            psnr = calc_PSNR(img1,img2)
            mse_skimage = skimage.measure.compare_mse(img1, img2)
            ssim_skimage = skimage.measure.compare_ssim(img1,img2,multichannel=True)
            full_ssim_skimage = skimage.measure.compare_ssim(img1,img2,multichannel=True,full=True)

            save_result(csv_dir,fullpath,psnr,mse_skimage,ssim_skimage)


    print(psnr)
    print(mse_skimage)
    print(ssim_skimage)

    print("{} files' PSNR Calculated, Please find the result in the folder.".format(len(filenames)))




if __name__=='__main__':
    main()

