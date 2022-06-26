import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from time import time
import random
import os
import numpy as np






def set_seed(seed):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)



##########################################

def conv_layer(in_ch, out_ch, kernel, pad_size):
    layer = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size = kernel, padding = pad_size),
        nn.BatchNorm2d(num_features = out_ch),
        nn.ReLU()
    )
    return layer

def conv_block(in_list, out_list, kernel_list, padding_list):
    layers = [conv_layer(in_list[i], out_list[i],kernel_list[i], padding_list[i]) for i in range(len(in_list))]
    return nn.Sequential(*layers)

def deconv_layer(in_ch, out_ch, kernel, pad_size):
    layer = nn.Sequential(
        nn.Conv2d(in_ch,out_ch,kernel_size = kernel,padding = pad_size),
        nn.BatchNorm2d(num_features = out_ch),
        nn.ReLU()
    )
    return layer

def deconv_block(in_list, out_list, kernel_list, padding_list):
    layers = [deconv_layer(in_list[i], out_list[i],kernel_list[i], padding_list[i]) for i in range(len(in_list))]   
    return nn.Sequential(*layers)




class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = True) # 256->128
        self.conv_layers_0 = conv_block([3,64,64],[64,64,64],[3,3,3],[1,1,1]) 
        self.conv_layers_1 = conv_block([64,128,128], [128,128,128], [3,3,3], [1,1,1]) 
        self.conv_layers_2 = conv_block([128,256], [256,256], [3,3], [1,1]) 
        self.conv_layers_3 = conv_block([256,512], [512,512], [3,3], [1,1]) 

        # bottleneck
        self.bottleneck_conv = conv_block([512,32,32],[32,32,512],[1,3,1],[0,1,0])

        # decoder
        self.unpool = nn.MaxUnpool2d(kernel_size = 2, stride = 2) # 16->32
        self.deconv_layer_3 = deconv_block([1024,512],[512,256],[3,3],[1,1]) 
        self.deconv_layer_2 = deconv_block([512,256],[256,128],[3,3],[1,1]) 
        self.deconv_layer_1 = deconv_block([256,128,128], [128,128,64], [3,3,3],[1,1,1]) 
        self.deconv_layer_0 = deconv_block([128,64,64], [64,64,1], [3,3,3],[1,1,1]) 

    def forward(self, x):
        # encoder
        e0 = self.conv_layers_0(x)
        e0p, ind_0 = self.pool(e0)
        e1 = self.conv_layers_1(e0p)
        e1p, ind_1 = self.pool(e1)
        e2 = self.conv_layers_2(e1p)
        e2p, ind_2 = self.pool(e2)
        e3 = self.conv_layers_3(e2p)
        e3p, ind_3 = self.pool(e3)

        # bottleneck
        b = self.bottleneck_conv(e3p)

        # decoder
        b = self.unpool(b, ind_3)
        d3 = self.unpool(self.deconv_layer_3(torch.cat((e3,b), axis = 1)), ind_2)
        d2 = self.unpool(self.deconv_layer_2(torch.cat((e2,d3), axis = 1)), ind_1)
        d1 = self.unpool(self.deconv_layer_1(torch.cat((e1,d2), axis = 1)), ind_0)
        d0 = self.deconv_layer_0(torch.cat((e0,d1),axis = 1)) # no activation
        return d0

