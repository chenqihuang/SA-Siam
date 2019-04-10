# -*- coding:utf-8 -*-
# !/ussr/bin/env python2
__author__ = "QiHuangChen"
import torch
import torch.nn as nn
import torch.nn.functional as F
from Train.train_config import *
import numpy as np
from torch.autograd import Variable
import re
import tqdm
from scipy import io as sio

# channel attention ,need to be alert
class Channel_attention_net(nn.Module):

    def __init__(self, channel=256, reduction=16):
        super(Channel_attention_net, self).__init__()
        self.Max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, 9),
            nn.ReLU(inplace=True),
            nn.Linear(9, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y.expand_as(x)


        x1 = self.Max_pool(x[:, :, 0:7, 0:7])
        x2 = self.Max_pool(x[:, :, 8:13, 0:7])
        x3 = self.Max_pool(x[:, :, 13:21, 0:7])
        x4 = self.Max_pool(x[:, :, 0:7, 8:13])
        x5 = self.Max_pool(x[:, :, 8:13, 8:13])
        x6 = self.Max_pool(x[:, :, 13:21, 8:13])
        x7 = self.Max_pool(x[:, :, 0:7, 13:21])
        x8 = self.Max_pool(x[:, :, 8:13, 13:21])
        x9 = self.Max_pool(x[:, :, 13:21, 13:21])
        # The MLP module shares weights across channels extracted from the same        convolutional        layer.



class AlexNet_like(nn.Module):

    def __init__(self):
        super(AlexNet_like, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),# norm before the activation
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))


        self.load_params_from_mat("/home/esc/Experiment/Code/SiamFC-PyTorch-master_HengLan_181030/models/pretrain_model_mat/imagenet-matconvnet-alex.mat")

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        return conv4, conv5


    def load_matconvnet(self, net_path):
        mat = sio.loadmat(net_path)
        params = mat['params']

        params = params[0]
        params_names = params['name']  # get net/params/name
        params_names_list = [params_names[p][0] for p in range(params_names.size)]
        params_names_list = params_names_list[0:10]
        params_values = params['value']  # get net/params/val
        params_values_list = [params_values[p] for p in range(params_values.size)]
        params_values_list = params_values_list[0:10]
        return params_names_list, params_values_list


    def load_params_from_mat(self, net_path):
        params_names_list, params_values_list = self.load_matconvnet(net_path)
        params_values_list = [torch.from_numpy(p) for p in params_values_list]  # values convert numpy to Tensor

        for index, param in enumerate(params_values_list):
            param_name = params_names_list[index]
            if 'conv' in param_name and param_name[-1] == 'f':
                param = param.permute(3, 2, 0, 1)
            param = torch.squeeze(param)
            params_values_list[index] = param

        self.conv1[0].weight.data[:] = params_values_list[params_names_list.index('conv%df' % 1)]
        self.conv1[0].bias.data[:] = params_values_list[params_names_list.index('conv%db' % 1)]
        print self.conv1[0].bias.data[:]

        self.conv2[0].weight.data[:] = params_values_list[params_names_list.index('conv%df' % 2)]
        self.conv2[0].bias.data[:] = params_values_list[params_names_list.index('conv%db' % 2)]

        self.conv3[0].weight.data[:] = params_values_list[params_names_list.index('conv%df' % 3)]
        self.conv3[0].bias.data[:] = params_values_list[params_names_list.index('conv%db' % 3)]

        self.conv4[0].weight.data[:] = params_values_list[params_names_list.index('conv%df' % 4)]
        self.conv4[0].bias.data[:] = params_values_list[params_names_list.index('conv%db' % 4)]

        self.conv5[0].weight.data[:] = params_values_list[params_names_list.index('conv%df' % 5)]
        self.conv5[0].bias.data[:] = params_values_list[params_names_list.index('conv%db' % 5)]


class S_Net(nn.Module):

    def __init__(self):
        super(S_Net, self).__init__()
        self.feat_extraction = AlexNet_like()
        for m in self.feat_extraction.parameters():
            m.requires_grad = False

        self.conv4_att = Channel_attention_net()
        self.conv5_att = Channel_attention_net()
        self.fuse_conv4 = nn.Conv2d(384, 128, 1, 1)
        self.fuse_conv5 = nn.Conv2d(256, 128, 1, 1)

        self.adjust = nn.Conv2d(1, 1, 1, 1)
        self._initialize_weight()


    def forward(self, z, x):
        z_feat4, z_feat5 = self.feat_extraction(z)
        x_feat4, x_feat5 = self.feat_extraction(x)
        # attention
        z_feat4_attention =self.conv4_att(z_feat4)
        z_feat5_attention =self.conv5_att(z_feat5)
        # crop layer
        z_feat4 = z_feat4[:, :, 9:14, 9:14]
        z_feat5 = z_feat5[:, :, 9:14, 9:14]
        #
        z_feat4 = z_feat4_attention * z_feat4
        z_feat5 = z_feat5_attention * z_feat5
        z_feat4 = self.fuse_conv4(z_feat4)
        z_feat5 = self.fuse_conv5(z_feat5)
        #correlation
        xcorr_out4 = self.xcorr(z_feat4, x_feat4)
        xcorr_out5 = self.xcorr(z_feat5, x_feat5)
        xcorr_out = xcorr_out4 + xcorr_out5
        score = self.adjust(xcorr_out)
        return score


    def xcorr(self, z, x):
        """
        correlation layer as in the original SiamFC (convolution process in fact)
        """
        batch_size_x, channel_x, w_x, h_x = x.shape
        x = torch.reshape(x, (1, batch_size_x * channel_x, w_x, h_x))

        # group convolution
        out = F.conv2d(x, z, groups = batch_size_x)

        batch_size_out, channel_out, w_out, h_out = out.shape
        xcorr_out = torch.reshape(out, (channel_out, batch_size_out, w_out, h_out))

        return xcorr_out


    def _initialize_weight(self):
        pass

    def weight_loss(self, prediction, label, weight):
        """
        weighted cross entropy loss
        """
        return F.binary_cross_entropy_with_logits(prediction,
                                                  label,
                                                  weight,
                                                  size_average=False) / self.config.batch_size

if __name__ == "__main__":
    pass

