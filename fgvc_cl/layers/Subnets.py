import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch
from fgvc_cl.utils.Register import REGISTRY, REGISTER_MODEL

"""
By Fangrui Liu (fangrui.liu@ubc.ca) Zihao Liu
University of British Columbia Okanagan
Peking University
"""

class Colearning(nn.Module):
    def __init__(self, in_channels, num_class):
        super(Colearning, self).__init__()

        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels=2*in_channels, out_channels=512, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, stride=1),
        )

        self.aggre_clf = Aggre_clf(512, num_class)

    def forward(self, feature_ori, feature_extra):

        feature_cat = torch.cat([feature_ori, feature_extra], dim=1)
        feature_cat = self.convblock(feature_cat)

        ########classification of concat feature##################
        #######可能要加个Resnet自带的GAP
        logits = self.aggre_clf(feature_cat)

        return logits


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

# Squeeze and Excitation Block Module
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels * 2, 1, bias=False),
        )

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1) # Squeeze
        w = self.fc(w)
        w, b = w.split(w.data.size(1) // 2, dim=1) # Excitation，第1维进行间隔为w.data.size(1)的分块
        w = torch.sigmoid(w)

        return x * w + b # Scale and add bias

class Aggre_clf(nn.Module):
    def __init__(self, in_channels, num_class):
        super(Aggre_clf, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1)
        self.relu = nn.LeakyReLU()
        self.batchnorm1 = nn.BatchNorm2d(512)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.batchnorm2 = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1)

        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_class)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.batchnorm1(x1)

        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.batchnorm2(x1)

        x1 = self.conv3(x1)
        x1 = self.sigmoid(x1)

        final_x = x + x.mul(x1)
        final_x = final_x.flatten(1)

        logits = self.fc(final_x)

        return logits