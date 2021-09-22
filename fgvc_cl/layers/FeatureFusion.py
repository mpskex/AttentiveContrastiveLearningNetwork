from torch import nn
import math
from torch.nn import functional as F
import torch

class FeatureFusion(nn.Module):
    """
    Feature Fusion for every pair of features
    by Fangrui Liu @ University of British Columbia
    fangrui.liu@ubc.ca
    """
    def __init__(self, fusion_prob=0.2):
        super(FeatureFusion, self).__init__()
        self.fusion_prob = fusion_prob

    def forward(self, feature_att, feature_neg):
        #   (NCHW)
        #   (batch_size, atten_num, channel, *feature_size)
        batch_size = feature_att.size()[0]
        atten_num = feature_att.size()[1]
        indxs = torch.randint(0, atten_num, [batch_size, math.ceil(self.fusion_prob * atten_num)])
        feature_att[indxs] = feature_neg[indxs]
        return
