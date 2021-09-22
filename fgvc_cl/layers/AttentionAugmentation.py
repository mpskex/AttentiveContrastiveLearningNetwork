from torch import nn
from torch.nn import functional as F
import torch

class AttentionAugmentation(nn.Module):
    """
    Use the generated attention to augment the data
    by Fangrui Liu @ University of British Columbia
    fangrui.liu@ubc.ca
    """
    def __init__(self, num_attention):
        """
        :param num_attention:
        """
        self.num_attention = num_attention
        super(AttentionAugmentation, self).__init__()

    def forward(self, image, attentions, tolist=False):
        """
        forward the layer
        :param image:       (batch_size, channel, height, width)
        :param attentions:  (batch_size, attention_num, height, width)
        :param tolist:
        :return:
        """
        aug_img = []
        for idx, attention in enumerate(attentions.split(1, dim=0)):
            single_aug_img = []
            for at in attention.split(1, dim=1):
                at = at.expand(-1, 3, -1, -1)
                at = F.interpolate(at, size=image.size()[2:], align_corners=False, mode='bilinear')
                single_aug_img.append(at[0] * image[idx])
            if not tolist:
                single_aug_img = torch.stack(single_aug_img, dim=0)
            aug_img.append(single_aug_img)
        if not tolist:
            aug_img = torch.stack(aug_img, dim=0)
        return aug_img

if __name__ == '__main__':
    from fgvc_cl.utils.Profiler import statistic_track
    batch_size = 4
    num_attention = 16
    att_aug = AttentionAugmentation(16)
    wrapped = statistic_track(att_aug.forward, samples=10)
    augmented = wrapped(torch.rand(batch_size, 3, 448, 448), torch.rand(batch_size, num_attention, 8, 8))
    print(augmented.size())