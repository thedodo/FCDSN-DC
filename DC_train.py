import os
import sys
import glob
import numpy as np
from numpy import inf
import cv2
import re
import time
from torch import optim

import time
import itertools
import timeit
import argparse
import imutils
from PIL import Image

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

import skimage
import numpy.matlib

import matplotlib.pyplot as plt
import progressbar

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

import random
from termcolor import colored
import configparser

Tensor = torch.cuda.FloatTensor
LTensor = torch.cuda.LongTensor


def main():
    
    config = configparser.ConfigParser()
    config.read(sys.argv[1])  
    
    patch_size = int(config['PARAM']['patch_size'])
    nr_epochs = int(config['PARAM']['nr_epochs'])
    batch_size = int(config['PARAM']['batch_size'])
    model_name = config['PARAM']['model_name']
    s_range = int(config['PARAM']['s_range'])
    chan = int(config['PARAM']['chan'])
    dataset = config['PARAM']['dataset']
    n_conv = int(config['PARAM']['num_conv'])
    lr = float(config['PARAM']['lr'])
    
    disp_list = config['PARAM']['disp_list']
    gt_list = config['PARAM']['gt_list']
    im_left_list = config['PARAM']['im_left_list']
    out_folder = config['PARAM']['out_folder']
    w_folder = config['PARAM']['w_folder']   
    
    
    print("Patch size: ", patch_size)
    print("Number Epochs: ", nr_epochs)
    print("Batch-size: ", batch_size)
    print("Model name: ", model_name)
    print("Shift range: ", s_range)
    print("Number of Channels: ", chan)
    print("Dataset: ", dataset)
    
    
    disp_list_f = glob.glob(disp_list)
    disp_list_f = sorted(disp_list_f)
    
    gt_list_f = glob.glob(gt_list)
    gt_list_f = sorted(gt_list_f)
    
    im_left_list_f = glob.glob(im_left_list)
    im_left_list_f = sorted(im_left_list_f)
   
    
    updInc = createNW(chan,n_conv)
    
    nolabel_list, loss_func = getClassWeights(disp_list_f, gt_list_f,chan)
    loss_func = nn.CrossEntropyLoss()
    
    train(chan, dataset, updInc, loss_func, batch_size, nr_epochs,out_folder, disp_list_f, gt_list_f, im_left_list_f, patch_size, nolabel_list, w_folder,model_name,lr)    
    
        
def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)


def createNW(chan,n_conv):
    
    class UpdateInconsNW(nn.Module):
        def __init__(self,img_ch=chan):
            super(UpdateInconsNW,self).__init__()
            
            self.softmax = nn.Softmax(dim=1)
            
            self.act = nn.ReLU()
            
            self.Conv1 = nn.Conv2d(img_ch, n_conv, kernel_size = 3,stride=1,padding = 1,dilation = 1, bias=True)
            self.Conv2 = nn.Conv2d(n_conv, n_conv, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)        
            self.Conv3 = nn.Conv2d(n_conv + 3, img_ch, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)
            
            
        def forward(self,x_in, im):
            
            x1 = self.Conv1(x_in)
            x1 = self.act(x1)
                            
            x2 = self.Conv2(x1)
            x2 = self.act(x2)
                    
            x3im = torch.cat((x2,im),axis = 1)
            
            x3 = self.Conv3(x3im)
            x3 = self.softmax(x3)
            
            return x3
    
    updInc = UpdateInconsNW()
    updInc = updInc.cuda()
    
    upd_nw_params = sum(p.numel() for p in updInc.parameters() if p.requires_grad)
    print("Depth-Completion Network: " ,upd_nw_params)
    return updInc


def calcEPE(disp, gt_fn):
    
    gt = gt_fn

    gt[np.where(gt == np.inf)] = -100
    
    mask = gt > 0
    
    disp = np.squeeze(disp)
    disp = disp[mask]
    gt = gt[mask]

    nr_px = len(gt)

    abs_error_im = np.abs(disp - gt)

    five_pe = (float(np.count_nonzero(abs_error_im >= 5.0) ) / nr_px) * 100.0  
    four_pe = (float(np.count_nonzero(abs_error_im >= 4.0) ) / nr_px) * 100.0  
    three_pe = (float(np.count_nonzero(abs_error_im >= 3.0) ) / nr_px) * 100.0  
    two_pe = (float(np.count_nonzero(abs_error_im >= 2.0) ) / nr_px) * 100.0        
    one_pe = (float(np.count_nonzero(abs_error_im >= 1.0) ) / nr_px) * 100.0        
    pf_pe = (float(np.count_nonzero(abs_error_im >= 0.5) ) / nr_px) * 100.0  
        
    return five_pe, four_pe, three_pe, two_pe, one_pe, pf_pe

def findGTInDispArr(chan, arr, gt, offset):
    c,w,h = arr.shape
    first_arr = np.zeros((w,h))
    
    #TO SEE HOW MANY ARE STILL NOT USABLE!!!
    first_arr = first_arr * -1
    
    for w_ in range(0,w-1):
      for h_ in range(0,h-1):
        found = 0
        for i in range(0,chan):
            if (int(arr[i,w_,h_]) == (int(gt[w_,h_])+offset)):
                first_arr[w_,h_] = i
                found = 1
                break
        if(found == 0):
            for i in range(0,chan):
                if (int(arr[i,w_,h_]) == (int(gt[w_,h_])+offset+1)):
                    first_arr[w_,h_] = i
                    found = 1
                    break
        if(found == 0):
            for i in range(0,chan):
                if (int(arr[i,w_,h_]) == (int(gt[w_,h_])+offset-1)):
                    first_arr[w_,h_] = i
                    found = 1
                    break
        if(found == 0):
            for i in range(0,chan):
                if (int(arr[i,w_,h_]) == (int(gt[w_,h_])+offset+2)):
                    first_arr[w_,h_] = i
                    found = 1
                    break
        if(found == 0):
            for i in range(0,chan):
                if (int(arr[i,w_,h_]) == (int(gt[w_,h_])+offset-2)):
                    first_arr[w_,h_] = i
                    found = 1
                    break         
    return first_arr

def findGTInDispArrSingle(arr, gt, offset):
    
    first = -1
    #this loop could probably be sped up right?
    for i in range(0,len(arr)):
        if (int(arr[i]) == (int(gt)+offset)):
            first = i
            break
            
    return first


def loadMB(disp_list_f, gt_list_f, im_left_list_f):
    
    disp_list = []
    gt_list = []
    im_list = []
    names = []
          
    for i in range(0,len(disp_list_f)):
        
        cur_disp, _ = readPFM(disp_list_f[i])
        cur_gt, _ = readPFM(gt_list_f[i])
        
        cur_disp[np.isnan(cur_disp)] = 0
        cur_disp[np.isinf(cur_disp)] = 0

        cur_gt[np.isnan(cur_gt)] = 0
        cur_gt[np.isinf(cur_gt)] = 0
                    
        cur_im = cv2.imread(im_left_list_f[i])
        cur_im = (cur_im - np.min(cur_im)) / (np.max(cur_im) - np.min(cur_im))
        
        disp_list.append(cur_disp)
        gt_list.append(cur_gt)
        im_list.append(cur_im)
        names.append(disp_list_f[i].split('/')[-2])
        
    return disp_list, gt_list,im_list,names

def createShiftPytZero(image, chan):
    
    counter = np.ones((image.shape[0],image.shape[1])) * chan
    counterT = Variable(Tensor(counter))
    
    shift_arr = np.zeros((chan,image.shape[0],image.shape[1]))
    shift_arrT = Variable(Tensor(shift_arr))

    i = 0
    while(torch.sum(counterT) > 0):

        if(i == image.shape[1]):
            i = 0
            
        if(i % 2 == 0):                
            ex_s = torch.roll(image,-i)
            #ex_s = torch.roll(image,i)
            #set left side of tensor to zero to mimick old createshift
            ex_s[:,chan-i:chan] = 0
            
        if(i % 2 == 1):
            ex_s = torch.roll(image,i)
            #ex_s = torch.roll(image,-i)
            #set right side of tensor to zero to mimick old createshift
            ex_s[:,0:i] = 0
            
        
        idc = torch.nonzero(ex_s, as_tuple = True)

        counterT[idc[0],idc[1]] += -1
        
        max_loop = torch.min(counterT).cpu().data.numpy().astype(np.int)

        #it overwrites lines that already have values with 0's!!!
        for d in range(max_loop,chan):
            
            idx_cur = torch.where(counterT == d)
            slice_tensor = torch.zeros(ex_s.shape[0], ex_s.shape[1]).cuda()
            slice_tensor[idx_cur[0].long(),idx_cur[1].long()] = ex_s[idx_cur[0].long(),idx_cur[1].long()]
            
            idc_slice = torch.nonzero(slice_tensor, as_tuple = True)
            shift_arrT[d, idc_slice[0].long(),idc_slice[1].long()] = ex_s[idc_slice[0].long(),idc_slice[1].long()]
            
        counterT[counterT < 0] = 0
        i = i + 1
                    
    return shift_arrT


def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

#problem if upd_count == 0??
def getClass(disp_list_f, gt_list_f, chan):
    
    keep_list = []
    upd_list = []
    disp_list = []
    gt_list = []
    nolabel_list = []
    
    for i in range(0,len(disp_list_f)): 

        
        cur_disp, _ = readPFM(disp_list_f[i])
        cur_gt, _ = readPFM(gt_list_f[i])
        
        cur_disp[np.isnan(cur_disp)] = 0
        cur_disp[np.isinf(cur_disp)] = 0

        cur_gt[np.isnan(cur_gt)] = 0
        cur_gt[np.isinf(cur_gt)] = 0
        
        disp_list.append(cur_disp)
        gt_list.append(cur_gt)     
        
        h,w = cur_disp.shape

        cur_upd = np.zeros((h,w))
        cur_upd[np.where(cur_disp == 0)] = 1
            
        cur_keep = np.zeros((h,w))
        cur_keep[np.where(cur_upd == 0)] = 1

        
        keep_list.append(cur_keep)
        upd_list.append(cur_upd)    
    
    total_px_to_upd = 0
    for s in range(0,len(disp_list_f)): 
        
        #gets number of 0 and 1 pixels! need 1 pixels (to update)
        upd_class,upd_counts = np.unique(upd_list[s], return_counts = True)    
        #if upd_counts is not empty
        if(len(upd_counts) == 2):
            total_px_to_upd = total_px_to_upd + upd_counts[1]
    
    batch_gt = np.zeros((total_px_to_upd,1))

    #count for batch_gt array!
    el = 0
    nol_count = 0
    for c in range(0,len(disp_list_f)):
        
        print('------------------')
        print(disp_list_f[c])
        print(gt_list_f[c])
        disp = disp_list[c]    
        gt = gt_list[c]

        keep = keep_list[c]
        
        disp = disp.copy()
        dispT = Variable(Tensor(disp))
        
        nolabels = np.zeros((disp.shape[0], disp.shape[1]))
        
        cur_disp_shifted = createShiftPytZero(dispT,chan)
        h,w = gt.shape
        
        #only way to increase performance is to do this for the whole image at once, not per pixel!
        #i am sure it is possible but...
        for h_ in range(h):
            for w_ in range(w):
                
                if(keep[h_,w_] == 0):
                    d_arr = torch.squeeze(cur_disp_shifted[:,h_,w_])
                    cur_gt = gt[h_,w_]
                    cur_gt = int(np.round(cur_gt))

                    gt_label = findGTInDispArrSingle(d_arr, cur_gt, 0)  
                    if(gt_label == -1):
                        gt_label = findGTInDispArrSingle(d_arr, cur_gt, 1)
                        if(gt_label == -1):
                            gt_label = findGTInDispArrSingle(d_arr, cur_gt, -1)                        
                            if(gt_label == -1):
                                gt_label = findGTInDispArrSingle(d_arr, cur_gt, 2)
                                if(gt_label == -1):
                                    gt_label = findGTInDispArrSingle(d_arr, cur_gt, -2)                        

                    #maybe this is the reason! Maybe there are so many
                    #zeros because it does not find the labels?
                    #test here with -1!
                    if(int(gt[h_,w_]) > 0):
                        if(gt_label > -1):
                            batch_gt[el,0] = int(gt_label)
                            el = el + 1
                        else:
                            nolabels[h_,w_] = 1
                            nol_count = nol_count + 1
                            
        
        folder = gt_list_f[c].replace(gt_list_f[c].split('/')[-1],'')
        print(folder)
                    
        name = gt_list_f[c].split('/')[-1].split('.')[0]
        print(folder + name + 'no_labels.png')
            
        cv2.imwrite(folder + name + 'no_labels.png', nolabels * 255)                 
        
        nolabel_list.append(nolabels)
        print('------------------')
    
    print("nolabel count: {}".format(nol_count))
    del cur_disp_shifted
    del disp
    del gt
    
    return batch_gt, nolabel_list



def getClassWeights(disp_list_f, gt_list_f,chan):
    #should be vec with all classes over all disparities!!
    gtlabels2count,nolabel_list = getClass(disp_list_f, gt_list_f,chan)
    gt_classes, gt_counts = np.unique(gtlabels2count.astype(np.uint8), return_counts = True)
    
    #TODO: Remove pixels with invalid gt from dataset!!! (probably best to include that in keep, update)
    #TODO: RE-VISIT THIS! Does it work?
    if(len(gt_classes) < (chan)):
        for w_ in range(0,chan):
            if(w_ not in gt_classes):
                gt_counts = np.insert(gt_counts,w_ , 100000)    
    norm_w = []
    for w_ in range(0,chan):
        cur_w = 1 / gt_counts[w_]
        norm_w.append(cur_w)
    
    
    class_weights = torch.FloatTensor(norm_w).cuda()
    loss_func = nn.CrossEntropyLoss(weight=class_weights)
    
    return nolabel_list, loss_func


def saveClassWeights():
    norm_w,nolabel_list = getClassWeights()
    with open('mb2021.txt', 'w') as f:
        for item in norm_w:
            f.write("%s\n" % item)
    
    #load weights!
    #norm_w=[]
    #with open('kitti2015c10.txt', "r") as file1:
    #    for line in file1.readlines():
    #        norm_w.append(float(line))
    #
    ## Important: Convert Weights To Float Tensor


#how are there float values here????
#somehow wrong!
def TestMB(names, updInc, chan, disp_list, gt_list, im_list, out_folder,nr_iter, save):
    
    avg_two_pe = 0.0
    
    for i in range(len(disp_list)): 
        
        disp  = disp_list[i]
        gt = gt_list[i]
        im = im_list[i]
        name = names[i]
        
        h,w = disp.shape
        
        upd = np.zeros((h,w))
        upd[np.where(disp == 0)] = 1
            
        keep = np.zeros((h,w))
        keep[np.where(upd == 0)] = 1
        

        h,w,c = im.shape
        im = np.reshape(im, (c,h, w))
        im = im[np.newaxis,...]
        imT = Variable(Tensor(im.astype(np.uint8)))

        upd = np.zeros((disp.shape[0],disp.shape[1]))
        upd[np.where(disp == 0)] = 1

        keep = np.zeros((disp.shape[0],disp.shape[1]))
        keep[np.where(upd == 0)] = 1

        keep_t = disp * keep
        
        updT = Variable(Tensor(upd.astype(np.float32)))
        keepT_t = Variable(Tensor(keep_t.astype(np.float32)))
        dispT = Variable(Tensor(disp.astype(np.uint8)))
        dispShift = createShiftPytZero(dispT, chan)
        
        dispShift = dispShift.unsqueeze(0)
        
        OutT = updInc(dispShift,imT) 
        OutT = torch.squeeze(OutT)
        bs,c,x,y = dispShift.shape

        idc_for_updt = torch.argmax(OutT, axis=0).unsqueeze(0)  
        pred = torch.gather(np.squeeze(dispShift), 0, idc_for_updt).squeeze()

        updT_t = pred * updT

        final_outp = keepT_t + updT_t
        dispT = final_outp

        five_pe, four_pe, three_pe, two_pe, one_pe, pf_pe = calcEPE(dispT.cpu().data.numpy().astype(np.float32), gt.astype(np.float32))
        avg_two_pe = two_pe + avg_two_pe   
        
        if(save == True):
            writePFM(out_folder + name + '.pfm', final_outp.cpu().data.numpy().astype(np.float32))
            
        del dispT
        del idc_for_updt
        del updT_t
        del OutT
        del dispShift
        
        torch.cuda.empty_cache()
        
        
    avg_two_pe = avg_two_pe / len(disp_list)
    return avg_two_pe


#how are there float values here????
def TestMBRecurrent(out_folder,nr_iter, save):
    
    n_list = disp_list_f
    avg_five_pe = 0.0
    avg_four_pe = 0.0 
    avg_three_pe = 0.0 
    avg_two_pe = 0.0
    avg_one_pe = 0.0
    avg_pf_pe = 0.0
    nr_samples = len(disp_list_f)
    
    for i in range(len(disp_list_f)): 
        
        t = time.time()
        
        disp  = disp_list[i]
        gt = gt_list[i]
        im = im_list[i]

        h,w,c = im.shape
        im = np.reshape(im, (c,h, w))
        im = im[np.newaxis,...]        
        imT = Variable(Tensor(im.astype(np.uint8)))

        upd = np.zeros((disp.shape[0],disp.shape[1]))
        upd[np.where(disp == 0)] = 1

        keep = np.zeros((disp.shape[0],disp.shape[1]))
        keep[np.where(upd == 0)] = 1

        keep_t = disp * keep
        
        updT = Variable(Tensor(upd.astype(np.float32)))
        keepT_t = Variable(Tensor(keep_t.astype(np.float32)))
        dispT = Variable(Tensor(disp.astype(np.uint8)))
        
        for d in range(0,nr_iter):

            dispShift = createShiftPytZero(dispT)
            dispShift = dispShift.unsqueeze(0)
            
            OutT = updInc(dispShift,imT) 
            OutT = torch.squeeze(OutT)

            bs,c,x,y = dispShift.shape

            idc_for_updt = torch.argmax(OutT, axis=0).unsqueeze(0)  
            pred = torch.gather(np.squeeze(dispShift), 0, idc_for_updt).squeeze()
            
            updT_t = pred * updT
            
            final_outp = keepT_t + updT_t
            dispT = final_outp
            
            elapsed = time.time() - t
            print ("Time: {}".format(elapsed))
            
            del OutT
            del idc_for_updt
            del dispShift
            torch.cuda.empty_cache()  
            
            
        disp_arr = dispT.cpu().data.numpy().astype(np.float32)
        disp_arr = cv2.medianBlur(disp_arr,5)

        five_pe, four_pe, three_pe, two_pe, one_pe, pf_pe = calcEPE(disp_arr, gt.astype(np.float32))
        
        print("2-PE: {}".format(two_pe))
        
        avg_five_pe = avg_five_pe + five_pe
        avg_four_pe = avg_four_pe +  four_pe
        avg_three_pe = avg_three_pe + three_pe
        avg_two_pe = avg_two_pe + two_pe
        avg_one_pe = avg_one_pe + one_pe
        avg_pf_pe = avg_pf_pe + pf_pe        
        
        name = n_list[i].split('/')[-1].replace('_s.pfm', '')
        if(save == True):
            writePFM(out_folder + name +'.pfm' , disp_arr)
        
        f= open(out_folder + name +'.txt',"w+")   
        f.write("runtime " + str(elapsed))
            
    avg_four_pe = avg_four_pe / nr_samples
    avg_two_pe = avg_two_pe / nr_samples
    avg_one_pe = avg_one_pe / nr_samples
    avg_pf_pe = avg_pf_pe / nr_samples
    
    print("4-PE: {}".format(avg_four_pe))
    print("2-PE: {}".format(avg_two_pe))
    print("1-PE: {}".format(avg_one_pe))
    print("0.5-PE: {}".format(avg_pf_pe))
    return avg_two_pe


def getBatch(chan, gt_list, im_list, patch_size, nr_ex, disp_list, nolabel_list):
    
    batch_x = Variable(Tensor(np.zeros((nr_ex,chan,patch_size,patch_size))))
    
    batch_gt = np.zeros((nr_ex,patch_size,patch_size))
    batch_im = np.zeros((nr_ex,3,patch_size,patch_size))
    
    ridx = np.random.randint(0,len(disp_list),1)
        
    disp = disp_list[ridx[0]]    
    gt = gt_list[ridx[0]]
    
    im = im_list[ridx[0]]
    h,w,c = im.shape
    im = np.reshape(im, (c,h, w))
    
    h,w = disp.shape
    upd = np.zeros((h,w))
    upd[np.where(disp == 0)] = 1

    keep = np.zeros((h,w))
    keep[np.where(upd == 0)] = 1

        
    nolabel = nolabel_list[ridx[0]]
    dispT = Variable(Tensor(disp.astype(np.uint8)))
    
    cur_disp_shifted = createShiftPytZero(dispT,chan)
    
    ps_h = int(patch_size/2)

    h,w = gt.shape
    
    for el in range(nr_ex):
        #get random position
        c,h,w = cur_disp_shifted.shape
        r_h = 0
        r_w = 0
        d = 0
        
        r_h = random.sample(range(0,h), 1)
        r_w = random.sample(range(0,w),1)
        d_arr = torch.squeeze(cur_disp_shifted[:,r_h[0]-ps_h:r_h[0]+(ps_h+1),r_w[0]-ps_h:r_w[0]+(ps_h+1)])
        cur_gt = gt[r_h[0]-ps_h:r_h[0]+(ps_h+1),r_w[0]-ps_h:r_w[0]+(ps_h+1)]
        cur_gt = cur_gt.astype(np.uint8)
        
        i = 0
        
        while(True): 
            r_h = random.sample(range(0,h), 1)
            r_w = random.sample(range(0,w), 1)
            i = i + 1
            if(r_h[0]-ps_h > 0):
              if(r_h[0]+ps_h < h):
                if(r_w[0]-ps_h > 0):
                  if(r_w[0]+ps_h < w):
                    if(int(gt[r_h[0],r_w[0]]) > 0):
                        if(keep[r_h[0],r_w[0]] == 0):
                            if(nolabel[r_h[0],r_w[0]] == 0):
                          
                                d_arr = torch.squeeze(cur_disp_shifted[:,r_h[0]-ps_h:r_h[0]+(ps_h+1),r_w[0]-ps_h:r_w[0]+(ps_h+1)])
                                cur_gt = gt[r_h[0]-ps_h:r_h[0]+(ps_h+1),r_w[0]-ps_h:r_w[0]+(ps_h+1)]
                                cur_gt = cur_gt.astype(np.uint8)
                                gt_label = findGTInDispArr(chan, d_arr.cpu().data.numpy().astype(np.float32), cur_gt, 0)
                                break
        
        cur_disp = cur_disp_shifted[:,r_h[0]-ps_h:r_h[0]+(ps_h+1),r_w[0]-ps_h:r_w[0]+(ps_h+1)]
        batch_x[el,:,:,:] = cur_disp       
        batch_gt[el,:,:] = gt_label

        batch_im[el,:,:,:] = im[:,r_h[0]-ps_h:r_h[0]+(ps_h+1),r_w[0]-ps_h:r_w[0]+(ps_h+1)]
        
    return batch_x, batch_gt, batch_im



def train(chan, dataset, updInc, loss_func, batch_size, nr_epochs,out_folder, disp_list_f, gt_list_f, im_left_list_f, patch_size, nolabel_list, w_folder,model_name, lr):
    
    if(dataset == 'MB' or dataset == 'MB2021'):
        disp_list, gt_list, im_list, names = loadMB(disp_list_f, gt_list_f, im_left_list_f)
    
    optimizer_G = optim.Adam(updInc.parameters(),  lr)
       
    best_two_pe = 100
    
    dispShift, gt, im = getBatch(chan, gt_list, im_list, patch_size, batch_size, disp_list, nolabel_list)
    
    imT = Variable(Tensor(im.astype(np.uint8)))
    gtT = Variable(LTensor(gt.astype(np.uint8)))
    
    
    for i in range(nr_epochs):
        
        #reset gradients
        optimizer_G.zero_grad()
        OutT = updInc(dispShift,imT) 
                
        loss = loss_func(OutT, gtT)
        loss = torch.mean(loss)
    
        loss.backward()
        optimizer_G.step()
    
        save = 1000
    
        if(i % save == 0):            
            
            print("EPOCH: {} CE-loss: {}".format(i,loss))  
            #probably does backprop!
            #avg_two_pe = TestMBRecurrent(out_folder,i, False)
            avg_two_pe = TestMB(names, updInc, chan, disp_list, gt_list, im_list, out_folder,1, False)
            
            print("2-PE Depth-Completion: {}".format(avg_two_pe))
    
            if(avg_two_pe < best_two_pe):
                
                avg_two_pe = TestMB(names, updInc, chan, disp_list, gt_list, im_list, out_folder,i, True)
                print(colored('------------------', 'green', attrs=['bold']))                         
                print(colored('NEW PB network: {}'.format(avg_two_pe), 'green', attrs=['bold']))  
                print(colored('------------------', 'green', attrs=['bold']))                         
    
                best_two_pe = avg_two_pe
                torch.save(updInc.state_dict(), w_folder + model_name + '_%06i' %(i) + 'e%06f' %(best_two_pe)) 
    
            dispShift, gt, im = getBatch(chan, gt_list, im_list, patch_size, batch_size, disp_list, nolabel_list)
            
            imT = Variable(Tensor(im.astype(np.uint8)))
            gtT = Variable(LTensor(gt.astype(np.uint8)))
            

if __name__ == "__main__":
    main()    