"""
(c) DI Dominik Hirner BSc. 
Institute for graphics and vision (ICG)
University of Technology Graz, Austria
e-mail: dominik.hirner@icg.tugraz.at
"""

import sys
import glob
import numpy as np
import cv2
import re
from termcolor import colored
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

import random
import argparse
import configparser
from typing import Tuple

from guided_filter_pytorch.guided_filter import GuidedFilter


def main():
    
    config = configparser.ConfigParser()
    
    config.read(sys.argv[1])  
    
    #continue from one of our trained weights
    transfer_train = config.getboolean('PARAM','transfer_train')
    #KITTI, MB or ETH
    dataset = config['PARAM']['dataset']
    #used as prefix for saved weights
    model_name = config['PARAM']['model_name']
    
    #folder with training data
    input_folder = config['PARAM']['input_folder']
    save_folder_branch = config['PARAM']['save_folder_branch']
    save_folder_simb = config['PARAM']['save_folder_simb']
    out_folder = config['PARAM']['out_folder']
    
    lr = float(config['PARAM']['lr'])
    
    batch_size = int(config['PARAM']['batch_size'])
    nr_batches = int(config['PARAM']['nr_batches'])
    nr_epochs = int(config['PARAM']['nr_epochs'])
    num_feat_branch = int(config['PARAM']['num_feat_branch'])
    num_feat_simb = int(config['PARAM']['num_feat_simb'])
    
    save_weights = int(config['PARAM']['save_weights'])
    
    #needs to be odd
    #size of patch-crops fed into the networ
    patch_size = int(config['PARAM']['patch_size'])
        
    #range for offset of o_neg
    r_low = int(config['PARAM']['r_low'])
    r_high = int(config['PARAM']['r_high'])
    
    print("Transfer train: ", transfer_train)
    print("Dataset: ", dataset)
    print("Model name: ", model_name)
    print("Input folder: ", input_folder)
    print("Batch-size: ", batch_size)
    print("learning rate: ", lr)
    print("Number of Epochs: ", nr_epochs)
    print("Patch size: ", patch_size)
    print("r_low: ", r_low)
    print("r_high: ", r_high)
    print("#Feature-maps for feature extractor per layer: ", num_feat_branch)
    print("#Feature-maps for similarity function per layer: ", num_feat_simb)
    print("Save weights every epochs: ", save_weights)        
    
    
    branch, simB = createNW(num_feat_branch, num_feat_simb)
    
    train(branch, simB,lr,input_folder, nr_epochs, nr_batches, batch_size, patch_size, dataset, out_folder,save_weights, save_folder_branch, save_folder_simb, model_name)
    

def disparity_sintel(filename):
    """ Return disparity read from filename. """
    f_in = np.array(Image.open(filename))
    d_r = f_in[:,:,0].astype('float64')
    d_g = f_in[:,:,1].astype('float64')
    d_b = f_in[:,:,2].astype('float64')

    depth = d_r * 4 + d_g / (2**6) + d_b / (2**14)
    return depth


def createNW(num_feat_branch, num_feat_simb):
    class DeformConv2D(nn.Module):
        def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
            super(DeformConv2D, self).__init__()
            self.kernel_size = kernel_size
            self.padding = padding
            self.zero_padding = nn.ZeroPad2d(padding)
            self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
    
        def forward(self, x, offset):
            dtype = offset.data.type()
            ks = self.kernel_size
            N = offset.size(1) // 2
    
            # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
            # Codes below are written to make sure same results of MXNet implementation.
            # You can remove them, and it won't influence the module's performance.
    
            if self.padding:
                x = self.zero_padding(x)
    
            # (b, 2N, h, w)
            p = self._get_p(offset, dtype)
    
            # (b, h, w, 2N)
            p = p.contiguous().permute(0, 2, 3, 1)
            q_lt = Variable(p.data, requires_grad=False).floor()
            q_rb = q_lt + 1
    
            q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
            q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
            q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
            q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)
    
            # (b, h, w, N)
            mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                              p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
            mask = mask.detach()
            floor_p = p - (p - torch.floor(p))
            p = p*(1-mask) + floor_p*mask
            p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
    
            # bilinear kernel (b, h, w, N)
            g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
            g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
            g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
            g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
    
            # (b, c, h, w, N)
            x_q_lt = self._get_x_q(x, q_lt, N)
            x_q_rb = self._get_x_q(x, q_rb, N)
            x_q_lb = self._get_x_q(x, q_lb, N)
            x_q_rt = self._get_x_q(x, q_rt, N)
    
            # (b, c, h, w, N)
            x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                       g_rb.unsqueeze(dim=1) * x_q_rb + \
                       g_lb.unsqueeze(dim=1) * x_q_lb + \
                       g_rt.unsqueeze(dim=1) * x_q_rt
    
            x_offset = self._reshape_x_offset(x_offset, ks)
            out = self.conv_kernel(x_offset)
    
            return out
    
        def _get_p_n(self, N, dtype):
            p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                              range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
            # (2N, 1)
            p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
            p_n = np.reshape(p_n, (1, 2*N, 1, 1))
            p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)
    
            return p_n
    
        @staticmethod
        def _get_p_0(h, w, N, dtype):
            p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
            p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
            p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
            p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
            p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)
    
            return p_0
    
        def _get_p(self, offset, dtype):
            N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
    
            # (1, 2N, 1, 1)
            p_n = self._get_p_n(N, dtype)
            # (1, 2N, h, w)
            p_0 = self._get_p_0(h, w, N, dtype)
            p = p_0 + p_n + offset
            return p
    
        def _get_x_q(self, x, q, N):
            b, h, w, _ = q.size()
            padded_w = x.size(3)
            c = x.size(1)
            # (b, c, h*w)
            x = x.contiguous().view(b, c, -1)
    
            # (b, h, w, N)
            index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
            # (b, c, h*w*N)
            index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
    
            x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
    
            return x_offset
    
        @staticmethod
        def _reshape_x_offset(x_offset, ks):
            b, c, h, w, N = x_offset.size()
            x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
            x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
    
            return x_offset
        
    class SiameseBranch64(nn.Module):
        def __init__(self,img_ch=3):
            super(SiameseBranch64,self).__init__()
            
            self.Tanh = nn.Tanh() 
            self.Conv1 = nn.Conv2d(img_ch, num_feat_branch, kernel_size = 5,stride=1,padding = 2,dilation = 1, bias=True)      
            self.Conv2 = nn.Conv2d(num_feat_branch, num_feat_branch, kernel_size = 5,stride=1,padding = 2,dilation = 1, bias=True)
            self.Conv3 = nn.Conv2d(2*num_feat_branch, num_feat_branch, kernel_size = 5,stride=1,padding = 2,dilation = 1, bias=True)
            self.Conv4 = nn.Conv2d(3*num_feat_branch, num_feat_branch, kernel_size = 5,stride=1,padding = 2,dilation = 1,bias=True)  
            
            
        def forward(self,x_in):
    
            x1 = self.Conv1(x_in) 
            x1 = self.Tanh(x1)
                    
            x2 = self.Conv2(x1) 
            x2 = self.Tanh(x2)
            
            d2 = torch.cat((x1,x2),dim=1)
            
            x3 = self.Conv3(d2) 
            x3 = self.Tanh(x3)
            
            d3 = torch.cat((x1,x2,x3),dim=1)
            
            x4 = self.Conv4(d3)
            
            return x4
        
    branch = SiameseBranch64()
    branch = branch.cuda()
    
    class SimMeasTanh(nn.Module):
        def __init__(self,img_ch=2*num_feat_branch):
            super(SimMeasTanh,self).__init__()
            
            self.tanh = nn.Tanh() 
            
            self.Conv1 = nn.Conv2d(img_ch, num_feat_simb, kernel_size = 3,stride=1,padding = 1,dilation = 1, bias=True)
            self.Conv2 = nn.Conv2d(num_feat_simb, num_feat_simb, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)
            self.Conv3 = nn.Conv2d(2*num_feat_simb, num_feat_simb, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)
            self.Conv4 = nn.Conv2d(3*num_feat_simb, num_feat_simb, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)
            self.Conv5 = nn.Conv2d(4*num_feat_simb, 1, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)
            
            self.conv_offset = nn.Conv2d(1, 18, kernel_size=3, padding=1, bias=None)
            self.deform_conv = DeformConv2D(1, 1, padding=1)
            
        def forward(self,x_in):
            
            x1 = self.Conv1(x_in) 
            x1 = self.tanh(x1)
                    
            x2 = self.Conv2(x1) 
            x2 = self.tanh(x2)
            
            d1 = torch.cat((x1,x2),dim=1)
    
            
            x3 = self.Conv3(d1) 
            x3 = self.tanh(x3)
            
            d2 = torch.cat((x1,x2,x3),dim=1)
            
            x4 = self.Conv4(d2) 
            x4 = self.tanh(x4) 
            d3 = torch.cat((x1,x2,x3,x4),dim=1)
            x5 = self.Conv5(d3)
            
            #needs to be positive for BCE!
            x5 = self.tanh(x5) 
            
            #deform_conv block!
            offsets = self.conv_offset(x5)
            x6 = self.deform_conv(x5,offsets)
            
            return x6
    
    simB = SimMeasTanh()
    simB = simB.cuda()
    
    return branch, simB


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


def loadETH3D(input_folder):
    
    left_filelist = glob.glob(input_folder + '/im0.png')
    right_filelist = glob.glob(input_folder + '/im1.png')
    disp_filelist = glob.glob(input_folder + '/disp0GT.pfm')
    
    left_filelist = sorted(left_filelist)
    right_filelist = sorted(right_filelist)
    disp_filelist = sorted(disp_filelist)
    
    left_list = []
    right_list = []
    disp_list = []
    
    for i in range(0,len(left_filelist)):
        
        cur_left = cv2.imread(left_filelist[i])
        cur_right = cv2.imread(right_filelist[i])
        cur_disp,_ = readPFM(disp_filelist[i])
        
        left_list.append(cur_left)
        right_list.append(cur_right)
        disp_list.append(cur_disp)
        
    return left_list, right_list, disp_list


def loadKitti2015(input_folder):

    left_filelist = glob.glob(input_folder + 'image_2/*.png')
    right_filelist = glob.glob(input_folder + 'image_3/*.png')
    disp_filelist = glob.glob(input_folder + 'disp_noc_0/*.png')
    
    left_filelist = sorted(left_filelist)
    right_filelist = sorted(right_filelist)
    disp_filelist = sorted(disp_filelist)

    left_elem_list = []
    for left_im in left_filelist:

        left_im_el = left_im.split('/')[-1]
        left_elem_list.append(left_im_el)

    left_elem_list = sorted(left_elem_list)

    right_elem_list = []
    for right_im in right_filelist:

        right_im_el = right_im.split('/')[-1]
        right_elem_list.append(right_im_el)

    right_elem_list = sorted(right_elem_list)

    gt_elem_list = []
    for gt_im in disp_filelist:

        gt_im_el = gt_im.split('/')[-1]
        gt_elem_list.append(gt_im_el)

    gt_elem_list = sorted(gt_elem_list)


    inters_list = set(left_elem_list) & set(right_elem_list) & set(gt_elem_list)
   
    inters_list = list(inters_list)
    left_list = []
    right_list = []
    disp_list = []
    
    for i in range(0,len(inters_list)):
        
        left_im = input_folder + 'image_2/' + inters_list[i]
        right_im = input_folder + 'image_3/' + inters_list[i]
        disp_im =  input_folder + 'disp_noc_0/' + inters_list[i] 
       
        cur_left = cv2.imread(left_im)
        cur_right = cv2.imread(right_im)
        cur_disp = cv2.imread(disp_im)
        
        cur_disp = np.mean(cur_disp,axis=2) 
        #set 0 (invalid) to inf to be same as MB for Batchloader
        cur_disp[np.where(cur_disp == 0.0)] = np.inf
        
        left_list.append(cur_left)
        right_list.append(cur_right)
        disp_list.append(cur_disp)
        
    return left_list, right_list, disp_list


def loadKitti2012(input_folder):

    left_filelist = glob.glob(input_folder + 'colored_0/*.png')
    right_filelist = glob.glob(input_folder + 'colored_1/*.png')
    disp_filelist = glob.glob(input_folder + 'disp_noc/*.png')
    
    left_filelist = sorted(left_filelist)
    right_filelist = sorted(right_filelist)
    disp_filelist = sorted(disp_filelist)

    left_elem_list = []
    for left_im in left_filelist:

        left_im_el = left_im.split('/')[-1]
        left_elem_list.append(left_im_el)

    left_elem_list = sorted(left_elem_list)

    right_elem_list = []
    for right_im in right_filelist:

        right_im_el = right_im.split('/')[-1]
        right_elem_list.append(right_im_el)

    right_elem_list = sorted(right_elem_list)

    gt_elem_list = []
    for gt_im in disp_filelist:

        gt_im_el = gt_im.split('/')[-1]
        gt_elem_list.append(gt_im_el)

    gt_elem_list = sorted(gt_elem_list)
    inters_list = set(left_elem_list) & set(right_elem_list) & set(gt_elem_list)
   
    inters_list = list(inters_list)
    left_list = []
    right_list = []
    disp_list = []
    
    for i in range(0,len(inters_list)):
        
        left_im = input_folder + 'colored_0/' + inters_list[i]
        right_im = input_folder + 'colored_1/' + inters_list[i]
        disp_im =  input_folder + 'disp_noc/' + inters_list[i] 
       
        cur_left = cv2.imread(left_im)
        cur_right = cv2.imread(right_im)
        cur_disp = cv2.imread(disp_im)
        
        cur_disp = np.mean(cur_disp,axis=2) 
        #set 0 (invalid) to inf to be same as MB for Batchloader
        cur_disp[np.where(cur_disp == 0.0)] = np.inf
        
        left_list.append(cur_left)
        right_list.append(cur_right)
        disp_list.append(cur_disp)
        
    return left_list, right_list, disp_list


def loadMB(input_folder):
    
    left_filelist = glob.glob(input_folder + '/im0.png')
    right_filelist = glob.glob(input_folder + '/im1.png')
    disp_filelist = glob.glob(input_folder + '/disp0GT.pfm')
    
    left_filelist = sorted(left_filelist)
    right_filelist = sorted(right_filelist)
    disp_filelist = sorted(disp_filelist)
    
    left_list = []
    right_list = []
    disp_list = []
    
    for i in range(0,len(left_filelist)):
        
        cur_left = cv2.imread(left_filelist[i])
        cur_right = cv2.imread(right_filelist[i])
        cur_disp,_ = readPFM(disp_filelist[i])
        
        cur_disp[np.isnan(cur_disp)] = 0
        cur_disp[np.isinf(cur_disp)] = 0
        
        left_list.append(cur_left)
        right_list.append(cur_right)
        disp_list.append(cur_disp)
        
    return left_list, right_list, disp_list


Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

def _compute_binary_kernel(window_size: Tuple[int, int]) -> torch.Tensor:
    r"""Creates a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: Tuple[int, ...] = tuple([(k - 1) // 2 for k in kernel_size])
    return computed[0], computed[1]


class MedianBlur(nn.Module):
    r"""Blurs an image using the median filter.

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = kornia.filters.MedianBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
    """

    def __init__(self, kernel_size: Tuple[int, int]) -> None:
        super(MedianBlur, self).__init__()
        self.kernel: torch.Tensor = _compute_binary_kernel(kernel_size)
        self.padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    def forward(self, input: torch.Tensor):  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # map the local window to single vector
        features: torch.Tensor = F.conv2d(
            input, kernel, padding=self.padding, stride=1, groups=c)
        features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

        # compute the median along the feature axis
        median: torch.Tensor = torch.median(features, dim=2)[0]
        return median

# functiona api
def median_blur(input: torch.Tensor,
                kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Blurs an image using the median filter.

    See :class:`~kornia.filters.MedianBlur` for details.
    """
    return MedianBlur(kernel_size)(input)


def filterCostVolMedianPyt(cost_vol):
    
    d,h,w = cost_vol.shape
    cost_vol = cost_vol.unsqueeze(0)
    
    for disp in range(d):

        cost_vol[:,disp,:,:] = median_blur(cost_vol[:,disp,:,:].unsqueeze(0), (5,5))
        
    return torch.squeeze(cost_vol)

def filterCostVolBilatpyt(cost_vol,left):
    
    left = np.mean(left,axis=2)
    leftT = Variable(Tensor(left))
    leftT = leftT.unsqueeze(0).unsqueeze(0)

    d,h,w = cost_vol.shape  
    
    f = GuidedFilter(8,10).cuda()  #10 #0.001
    
    for disp in range(d):
        cur_slice =  cost_vol[disp,:,:]
        cur_slice = cur_slice.unsqueeze(0).unsqueeze(0)
        
        inputs = [leftT, cur_slice]

        test = f(*inputs)
        cost_vol[disp,:,:] = np.squeeze(test)
        
    return cost_vol


#even further improve this by using pytorch!
def LR_Check(first_output, second_output):    
    
    h,w = first_output.shape
        
    line = np.array(range(0, w))
    idx_arr = np.matlib.repmat(line,h,1)    
    
    dif = idx_arr - first_output
    
    first_output[np.where(dif <= 0)] = 0
    
    first_output = first_output.astype(np.int)
    second_output = second_output.astype(np.int)
    dif = dif.astype(np.int)
    
    second_arr_reordered = np.array(list(map(lambda x, y: y[x], dif, second_output)))
    
    dif_LR = np.abs(second_arr_reordered - first_output)
    first_output[np.where(dif_LR >= 1.1)] = 0
    
    first_output = first_output.astype(np.float32)
    first_output[np.where(first_output == 0.0)] = np.nan
        
    return first_output

def createCostVol(branch, simB,left_im,right_im,max_disp):
        
    a_h, a_w,c = left_im.shape

    left_im = np.transpose(left_im, (2,0,1)).astype(np.uint8)
    right_im = np.transpose(right_im, (2,0,1)).astype(np.uint8)
    
    left_im = np.reshape(left_im, [1,c,a_h,a_w])
    right_im = np.reshape(right_im, [1,c,a_h,a_w])

    with torch.no_grad():

        left_imT = Variable(Tensor(left_im.astype(np.uint8)))
        right_imT = Variable(Tensor(right_im.astype(np.uint8)))

        left_feat = branch(left_imT)
        right_feat = branch(right_imT)
        
        _,f,h,w = left_feat.shape
        
        cost_vol = np.zeros((max_disp+1,a_h,a_w))
        cost_volT = Variable(Tensor(cost_vol))

        #0 => max_disp => one less disp!
        #python3 apparently cannot have 0 here for disp: right_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)  
        for disp in range(0,max_disp+1):

            if(disp == 0):
                
                sim_score = simB(torch.cat((left_feat, right_feat),dim=1))
                cost_volT[disp,:,:] = torch.squeeze(sim_score)                
            else:
                right_shifted = torch.cuda.FloatTensor(1,f,h,w).fill_(0)                      
                right_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)  
                right_appended = torch.cat([right_shift,right_feat],3)

                _,f,h_ap,w_ap = right_appended.shape
                right_shifted[:,:,:,:] = right_appended[:,:,:,:(w_ap-disp)]
                sim_score = simB(torch.cat((left_feat, right_shifted),dim=1))
                cost_volT[disp,:,:] = torch.squeeze(sim_score)              

    return cost_volT

def createCostVolRL(branch, simB, left_im,right_im,max_disp):

    a_h, a_w,c = left_im.shape

    left_im = np.transpose(left_im, (2,0,1)).astype(np.uint8)
    right_im = np.transpose(right_im, (2,0,1)).astype(np.uint8)
    
    left_im = np.reshape(left_im, [1,c,a_h,a_w])
    right_im = np.reshape(right_im, [1,c,a_h,a_w])

    with torch.no_grad():
        
        left_imT = Variable(Tensor(left_im))
        right_imT = Variable(Tensor(right_im))

        left_feat = branch(left_imT)
        right_feat = branch(right_imT)


        _,f,h,w = left_feat.shape
        cost_vol = np.zeros((max_disp+1,a_h,a_w))
        
        cost_volT = Variable(Tensor(cost_vol))

        for disp in range(0,max_disp+1):

            if(disp == 0):
                sim_score = simB(torch.cat((left_feat, right_feat),dim=1))
                cost_volT[disp,:,:] = torch.squeeze(sim_score) 
            else:    
                left_shifted = torch.cuda.FloatTensor(1,f,h,w).fill_(0)
                left_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)
                left_appended = torch.cat([left_feat,left_shift],3)

                _,f,h_ap,w_ap = left_appended.shape
                left_shifted[:,:,:,:] = left_appended[:,:,:,disp:w_ap]
            
                sim_score = simB(torch.cat((left_shifted, right_feat),dim=1))
                cost_volT[disp,:,:] = torch.squeeze(sim_score)
                
    return cost_volT

def TestImage(branch, simB, fn_left, fn_right, max_disp, filtered, lr_check):
    
    left = cv2.imread(fn_left)
    right = cv2.imread(fn_right)
    disp_map = []
    
    if(filtered):
        
        cost_vol = createCostVol(branch, simB, left,right,max_disp)
        
        cost_vol_filteredn = filterCostVolBilatpyt(cost_vol,left)
        cost_vol_filteredn = np.squeeze(cost_vol_filteredn.cpu().data.numpy())                
        disp = np.argmax(cost_vol_filteredn, axis=0) 
        
        del cost_vol
        del cost_vol_filteredn
        torch.cuda.empty_cache()              
        
        if(lr_check):
            cost_vol_RL = createCostVolRL(branch, simB,left,right,max_disp)
            
            cost_vol_RL_fn = filterCostVolBilatpyt(cost_vol_RL,right)
            cost_vol_RL_fn = np.squeeze(cost_vol_RL_fn.cpu().data.numpy())        
            
            disp_map_RL = np.argmax(cost_vol_RL_fn, axis=0)  
            disp_map = LR_Check(disp.astype(np.float32), disp_map_RL.astype(np.float32))
            
            
            del cost_vol_RL
            del cost_vol_RL_fn
            torch.cuda.empty_cache()              
        
    else:
        cost_vol = createCostVol(branch, simB,left,right,max_disp)
        cost_vol = np.squeeze(cost_vol.cpu().data.numpy())
        disp = np.argmax(cost_vol, axis=0)        
        
        if(lr_check):
            
            cost_vol_RL = createCostVolRL(branch, simB,left,right,max_disp)
            cost_vol_RL = np.squeeze(cost_vol_RL.cpu().data.numpy())
            disp_map_RL = np.argmax(cost_vol_RL, axis=0)       
            disp_map = LR_Check(disp.astype(np.float32), disp_map_RL.astype(np.float32))
    
    if(lr_check):
        return disp_map, disp, disp_map_RL
    else:
        return disp
    
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
    
def writePFMcyt(file, image, scale=1):
    file = open(file, 'wb')

    color = None

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

    scale = -scale

    file.write('%f\n'.encode() % scale)
    image.tofile(file)


def TestMB(branch, simB, input_folder, epoch, output_folder,filtered,lr_check,fill_incons, save):
    
    avg_five_pe = 0.0
    avg_four_pe = 0.0 
    avg_three_pe = 0.0 
    avg_two_pe = 0.0
    avg_one_pe = 0.0
    avg_pf_pe = 0.0
    algo_name = 'FC_sim_'

    nr_samples = len(glob.glob(input_folder))
    for samples in glob.glob(input_folder):

        
        gt,_ = readPFM(samples + 'disp0GT.pfm')

        f = open(samples + 'calib.txt','r')
        calib = f.read()
        max_disp = int(calib.split('\n')[6].split("=")[1])
        s_name = samples.split('/')[-2]

        disp = None
        disp_s = None

        if(lr_check):
            disp_s,disp, disp_rl = TestImage(branch, simB, samples + '/im0.png', samples + '/im1.png', max_disp, filtered, lr_check)
        else:
            disp = TestImage(branch, simB, samples + '/im0.png', samples + '/im1.png', max_disp, filtered, lr_check)

        disp = np.array(disp)

        gt = np.array(gt)
        
        five_pe, four_pe, three_pe, two_pe, one_pe, pf_pe = calcEPE(disp, gt)


        avg_five_pe = avg_five_pe + five_pe
        avg_four_pe = avg_four_pe +  four_pe
        avg_three_pe = avg_three_pe + three_pe
        avg_two_pe = avg_two_pe + two_pe
        avg_one_pe = avg_one_pe + one_pe
        avg_pf_pe = avg_pf_pe + pf_pe        

        if(save):
            writePFMcyt(output_folder + algo_name + s_name + '%06d.pfm' %epoch,disp.astype(np.float32))
            if(lr_check):
                writePFMcyt(output_folder + algo_name + s_name + '%06d_s.pfm' %epoch,disp_s) 
                writePFMcyt(output_folder + algo_name + s_name + '%06d_rl.pfm' %epoch,disp_rl.astype(np.float32)) 


    avg_four_pe = avg_four_pe / nr_samples
    avg_two_pe = avg_two_pe / nr_samples
    avg_one_pe = avg_one_pe / nr_samples
    avg_pf_pe = avg_pf_pe / nr_samples
    
    print("4-PE: {}".format(avg_four_pe))
    print("2-PE: {}".format(avg_two_pe))
    print("1-PE: {}".format(avg_one_pe))
    print("0.5-PE: {}".format(avg_pf_pe))
    
    return avg_two_pe


def TestKITTI2015(branch, simB,input_folder, epoch,output_folder,filtered,lr_check,fill_incons,save):
        
    avg_four_pe = 0.0 
    avg_two_pe = 0.0
    avg_pf_pe = 0.0 


    left_filelist = glob.glob(input_folder + 'image_2/*.png')
    right_filelist = glob.glob(input_folder + 'image_3/*.png')
    disp_filelist = glob.glob(input_folder + 'disp_noc_0/*.png')

    left_filelist = sorted(left_filelist)
    right_filelist = sorted(right_filelist)
    disp_filelist = sorted(disp_filelist)


    left_elem_list = []
    for left_im in left_filelist:

        left_im_el = left_im.split('/')[-1]
        left_elem_list.append(left_im_el)

    left_elem_list = sorted(left_elem_list)


    right_elem_list = []
    for right_im in right_filelist:

        right_im_el = right_im.split('/')[-1]
        right_elem_list.append(right_im_el)

    right_elem_list = sorted(right_elem_list)



    gt_elem_list = []
    for gt_im in disp_filelist:

        gt_im_el = gt_im.split('/')[-1]
        gt_elem_list.append(gt_im_el)

    gt_elem_list = sorted(gt_elem_list)


    inters_list = set(left_elem_list) & set(right_elem_list) & set(gt_elem_list)    
    inters_list = list(inters_list)

    #only test first 30 for time
    for i in range(0,30):  #len(inters_list)

        cur_gt = cv2.imread(input_folder + 'disp_noc_0/' +  inters_list[i])
        #RGB image
        cur_gt = np.mean(cur_gt, axis=2)

        cur_gt = cur_gt.astype(np.float32)
        cur_gt[np.where(cur_gt == 0)] = np.inf
        max_disp =  int(np.ceil(cur_gt[np.isfinite(cur_gt)].max())) + 1
        
        
        s_name = inters_list[i]
        disp = None
        
        if(lr_check):
            disp_s,disp, disp_rl = TestImage(branch, simB,input_folder + 'image_2/' + s_name, input_folder + 'image_3/' + s_name, max_disp, filtered, lr_check)   
        else:
            disp = TestImage(branch, simB,input_folder + 'image_2/' + s_name, input_folder + 'image_3/' + s_name, max_disp, filtered, lr_check) 
        
        disp = np.array(disp)

        gt = np.array(cur_gt)
        five_pe, four_pe, three_pe, two_pe, one_pe, pf_pe = calcEPE(disp, gt)

        avg_four_pe = avg_four_pe +  four_pe
        avg_two_pe = avg_two_pe + two_pe
        avg_pf_pe = avg_pf_pe + pf_pe        

        if(save):
            writePFMcyt(output_folder +  s_name + '%06d.pfm' %epoch,disp.astype(np.float32))
            if(lr_check):
                writePFMcyt(output_folder + s_name + '%06d_s.pfm' %epoch,disp_s) 
    
    avg_two_pe = avg_two_pe / (i+1)
    return avg_two_pe


def TestKITTI2012(branch, simB, input_folder, epoch, output_folder,filtered,lr_check,fill_incons, save):
    
        
    avg_four_pe = 0.0 
    avg_two_pe = 0.0
    avg_pf_pe = 0.0 


    left_filelist = glob.glob(input_folder + 'colored_0/*.png')
    right_filelist = glob.glob(input_folder + 'colored_1/*.png')
    disp_filelist = glob.glob(input_folder + 'disp_noc/*.png')

    left_filelist = sorted(left_filelist)
    right_filelist = sorted(right_filelist)
    disp_filelist = sorted(disp_filelist)


    left_elem_list = []
    for left_im in left_filelist:

        left_im_el = left_im.split('/')[-1]
        left_elem_list.append(left_im_el)

    left_elem_list = sorted(left_elem_list)


    right_elem_list = []
    for right_im in right_filelist:

        right_im_el = right_im.split('/')[-1]
        right_elem_list.append(right_im_el)

    right_elem_list = sorted(right_elem_list)



    gt_elem_list = []
    for gt_im in disp_filelist:

        gt_im_el = gt_im.split('/')[-1]
        gt_elem_list.append(gt_im_el)

    gt_elem_list = sorted(gt_elem_list)


    inters_list = set(left_elem_list) & set(right_elem_list) & set(gt_elem_list)    
    inters_list = list(inters_list)

    #only test first 30 for time
    for i in range(0,30):  #len(inters_list)

        cur_gt = cv2.imread(input_folder + 'disp_noc/' +  inters_list[i])
        #RGB image
        cur_gt = np.mean(cur_gt, axis=2)

        cur_gt = cur_gt.astype(np.float32)
        
        cur_gt[np.where(cur_gt == 0)] = np.inf
        max_disp =  int(np.ceil(cur_gt[np.isfinite(cur_gt)].max())) + 1
        
        
        s_name = inters_list[i]

        disp = None
        if(lr_check):
            disp_s,disp, disp_rl = TestImage(branch, simB, input_folder + 'colored_0/' + s_name, input_folder + 'colored_1/' + s_name, max_disp, filtered, lr_check)   
        else:
            disp = TestImage(branch, simB, input_folder + 'colored_0/' + s_name, input_folder + 'colored_1/' + s_name, max_disp, filtered, lr_check)   
        
        disp = np.array(disp)
        gt = np.array(cur_gt)
        
        five_pe, four_pe, three_pe, two_pe, one_pe, pf_pe = calcEPE(disp, gt)


        avg_four_pe = avg_four_pe +  four_pe
        avg_two_pe = avg_two_pe + two_pe
        avg_pf_pe = avg_pf_pe + pf_pe        

        if(save):
            writePFMcyt(output_folder +  s_name + '%06d.pfm' %epoch,disp.astype(np.float32))
            if(lr_check):
                writePFMcyt(output_folder + s_name + '%06d_s.pfm' %epoch,disp_s) 
    
    avg_two_pe = avg_two_pe / (i+1)
    return avg_two_pe

def TestETH(branch, simB, input_folder,epoch, output_folder,filtered,lr_check,fill_incons, save):
    
    avg_five_pe = 0.0
    avg_four_pe = 0.0 
    avg_three_pe = 0.0 
    avg_two_pe = 0.0
    avg_one_pe = 0.0
    avg_pf_pe = 0.0
    algo_name = 'FC_sim_'

    nr_samples = len(glob.glob(input_folder))
    for samples in glob.glob(input_folder):

        gt,_ = readPFM(samples + 'disp0GT.pfm')

        #f = open(samples + 'calib.txt','r')
        #calib = f.read()
        #take max_disp from gt!!
        #max_disp = int(calib.split('\n')[6].split("=")[1])
        max_disp =  int(np.ceil(gt[np.isfinite(gt)].max())) + 1
        s_name = samples.split('/')[-2]

        disp = None
        disp_s = None

        if(lr_check):
            disp_s,disp,disp_lr = TestImage(branch, simB,samples + '/im0.png', samples + '/im1.png', max_disp, filtered, lr_check)
        else:
            disp = TestImage(branch, simB,samples + '/im0.png', samples + '/im1.png', max_disp, filtered, lr_check)

        disp = np.array(disp)

        gt = np.array(gt)
        five_pe, four_pe, three_pe, two_pe, one_pe, pf_pe = calcEPE(disp, gt)

        avg_five_pe = avg_five_pe + five_pe
        avg_four_pe = avg_four_pe +  four_pe
        avg_three_pe = avg_three_pe + three_pe
        avg_two_pe = avg_two_pe + two_pe
        avg_one_pe = avg_one_pe + one_pe
        avg_pf_pe = avg_pf_pe + pf_pe        

        if(save):
            writePFMcyt(output_folder + algo_name + s_name + '%06d.pfm' %epoch,disp.astype(np.float32))
            if(lr_check):
                writePFMcyt(output_folder + algo_name + s_name + '%06d_s.pfm' %epoch,disp_s) 

    avg_two_pe = avg_two_pe / nr_samples
    return avg_two_pe


def calcEPE(disp, gt_fn):
    
    gt = gt_fn

    gt[np.where(gt == np.inf)] = -100   
    
    mask = gt > 0

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

#also RL patches => +d not -d!!
#needs to be odd
patch_size = 21

ps_h = int(patch_size/2)

#range for offset of o_neg
r_low = 1
r_high = 25

def getBatch(batch_size, patch_size, left_list, right_list, gt_list):
    
    batch_xl = np.zeros((batch_size,3,patch_size,patch_size))
    batch_xr_pos = np.zeros((batch_size,3,patch_size,patch_size))
    batch_xr_neg = np.zeros((batch_size,3,patch_size,patch_size))
    
    for el in range(batch_size):
        
        if(el % 25 == 0):
            
            ridx = np.random.randint(0,len(left_list),1)
            left_im = left_list[ridx[0]]
            right_im = right_list[ridx[0]]
            gt_im = gt_list[ridx[0]]
            
        #get random position
        h,w,c = left_im.shape
        r_h = 0
        r_w = 0
        d = 0
#        print('Draw for random position')
        #also check height! should not draw corner pixels!!
        while True:
            r_h = random.sample(range(ps_h,h-(ps_h+1)), 1)
            
            r_w = random.sample(range(ps_h,w-(ps_h+1)),1)   
            
            if(not np.isinf(gt_im[r_h,r_w])):
                d = int(np.round(gt_im[r_h,r_w]))
                if((r_w[0]-ps_h-d-1) >= 0):
                     if((r_w[0]+(ps_h+1)-d+1) <= w):
                        break
        
        d = int(np.round(gt_im[r_h,r_w]))
                
        cur_left = left_im[r_h[0]-ps_h:r_h[0]+(ps_h+1), r_w[0]-ps_h:r_w[0]+(ps_h+1),:]
        
        #choose offset
        o_pos = 0                
        cur_right_pos = right_im[r_h[0]-ps_h:r_h[0]+(ps_h+1), (r_w[0]-ps_h-d+o_pos):(r_w[0]+(ps_h+1)-d+o_pos),:]

        
        #should not be too close to real match!
        o_neg = 0
        while True:
            #range 6-8??? range(2,6)
            o_neg = random.sample(range(r_low,r_high), 1)
            if np.random.randint(-1, 1) == -1:
                o_neg = -o_neg[0]
            else:
                o_neg = o_neg[0]
            #try without d-+1   and(o_neg != (d-1)) and(o_neg != (d+1))
            if((o_neg != d) and ((r_w[0]-ps_h-d+o_neg) > 0)  and ((r_w[0]+(ps_h+1)-d+o_neg) < w)):
                break
        
        
        cur_right_neg = right_im[r_h[0]-ps_h:r_h[0]+(ps_h+1), (r_w[0]-ps_h-d+o_neg):(r_w[0]+(ps_h+1)-d+o_neg),:]

        
        batch_xl[el,:,:,:] =  np.transpose(cur_left, (2,0,1)).astype(np.uint8)
        batch_xr_pos[el,:,:,:] = np.transpose(cur_right_pos, (2,0,1)).astype(np.uint8)
        batch_xr_neg[el,:,:,:] = np.transpose(cur_right_neg, (2,0,1)).astype(np.uint8)
            
    return batch_xl, batch_xr_pos, batch_xr_neg


def my_hinge_loss(s_p, s_n):
    margin = 0.2
    relu = torch.nn.ReLU()
    relu = relu.cuda()
    loss = relu(-((s_p - s_n) - margin))

    return loss

def train(branch, simB,lr, input_folder, nr_epochs, nr_batches,batch_size, patch_size, dataset, out_folder,save_weights, save_folder_branch, save_folder_simb, model_name):
    
    params = list(branch.parameters()) + list(simB.parameters())
    if(dataset == 'MB'):
        left_list, right_list, gt_list = loadMB(input_folder)
    if(dataset == 'Kitti2012'):
        left_list, right_list, gt_list = loadKitti2012(input_folder)
    if(dataset == 'Kitti2015'):
        left_list, right_list, gt_list = loadKitti2015(input_folder)
    if(dataset == 'ETH'):
        left_list, right_list, gt_list = loadETH3D(input_folder)
    
    
    optimizer_G = optim.Adam(params, lr) 
    
    best_err = 100
    
    for i in range(nr_epochs):
        epoch_loss = 0.0
        for cur_batch in range(nr_batches): 
                
            batch_xl, batch_xr_pos, batch_xr_neg = getBatch(batch_size, patch_size, left_list, right_list, gt_list)
            
            #reset gradients
            optimizer_G.zero_grad()
    
            bs, c, h, w = batch_xl.shape
            batch_loss = 0.0
    
            xl = Variable(Tensor(batch_xl.astype(np.uint8)))
            xr_pos = Variable(Tensor(batch_xr_pos.astype(np.uint8)))
            xr_neg = Variable(Tensor(batch_xr_neg.astype(np.uint8)))
    
            left_out = branch(xl)
            right_pos_out = branch(xr_pos)
            right_neg_out = branch(xr_neg)
            
            sp = simB(torch.cat((left_out, right_pos_out),dim=1))
            sn = simB(torch.cat((left_out, right_neg_out),dim=1))
           
            
            batch_loss = my_hinge_loss(sp, sn)
            batch_loss = batch_loss.mean()
            
            batch_loss.backward()
            optimizer_G.step()
            
            epoch_loss = epoch_loss + batch_loss
    
        epoch_loss = epoch_loss/nr_batches
        if(i % save_weights == 0):
            print("EPOCH: {} loss: {}".format(i,epoch_loss))
    
            if(dataset == 'MB' or dataset == 'MB2021'):
                avg_2PE = TestMB(branch, simB, input_folder, 0,out_folder,False,False,False, True)
            if(dataset == 'Kitti2012'):
                avg_2PE = TestKITTI2012(branch, simB, input_folder, 0,out_folder,False,False,False, True)
            if(dataset == 'Kitti2015'):
                avg_2PE = TestKITTI2015(branch, simB, input_folder, 0,out_folder,False,False,False, True)
            if(dataset == 'ETH'):
                avg_2PE = TestETH(branch, simB, input_folder, 0,out_folder,False,False,False, True)
    
            if(avg_2PE < best_err):
                print(colored("NEW BP: {}".format(avg_2PE), 'green', attrs=['bold']))
                torch.save(branch.state_dict(), save_folder_branch + model_name + '_best%04i' %(i) + 'e%04f' %(avg_2PE)) 
                torch.save(simB.state_dict(), save_folder_simb + model_name + '_best%04i' %(i) + 'e%04f' %(avg_2PE)) 
                best_err = avg_2PE
            else:
                print("got worse")
                print(avg_2PE)

if __name__ == "__main__":
    main()    