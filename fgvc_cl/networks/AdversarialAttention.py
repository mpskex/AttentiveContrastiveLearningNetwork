import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch
from itertools import chain

from fgvc_cl.utils.Register import REGISTRY, REGISTER_MODEL, print_loaded
from fgvc_cl.layers.AttentionAugmentation import AttentionAugmentation
from fgvc_cl.layers.AttentionExperience import AttentionExperience
from fgvc_cl.layers.FeatureFusion import FeatureFusion
from fgvc_cl.layers.Subnets import PAM_Module, CAM_Module, SEBlock
from fgvc_cl.utils.Tricks import split_weights, xavier_init, LSR
"""
By Fangrui Liu (fangrui.liu@ubc.ca) Zihao Liu(lzh19961031@pku.edu.cn)
University of British Columbia Okanagan
Peking University
"""


class AttentionGenerator(nn.Module):
    def __init__(self, in_channel, num_attention):
        super().__init__()

        #self.seblock = SEBlock(in_channel)
        #self.sc = CAM_Module(in_channel)
        self.sa = PAM_Module(in_channel)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 512, 3, 1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, 1, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #   (batch_size, num_attention, *feature_size)
            #   for every attention, mask the input
            nn.Conv2d(256, num_attention, 1, bias=True),
        )

    def forward(self, x):
        #feature = self.seblock(x) + x  # recalibration先学channel间关系
        #feature = F.relu(feature)
        feature = self.sa(x)
        output = self.conv(x)
        return F.softmax(output, dim=1)


class AttentionRegularizer(nn.Module):
    #   TODO (zihao): fix this
    def __init__(self):
        super(AttentionRegularizer, self).__init__()
        self.criterion2 = nn.SmoothL1Loss()

    def forward(self, attention, feature_ori):
        #   Component #1: Attention from different channels should be different
        #   Component #2: Channel-wise attention should be aligned with the original feature
        coordinate = F.adaptive_avg_pool2d(attention, 1)
        center_coordinate = F.adaptive_avg_pool2d(feature_ori, 1)
        #   Global Average Pooling + L2 Loss
        loss = F.nll_loss(coordinate, center_coordinate)
        return loss


class FeatureConcat(nn.Module):
    def __init__(self):
        super(FeatureConcat, self).__init__()

    def forward(self, feature_ori, feature_aug):
        feature_aug_mean = feature_aug.mean(dim=1)
        feature_merge = torch.cat([feature_ori, feature_aug_mean], dim=1)
        return feature_merge


class PartClassifier(nn.Module):
    def __init__(self, feature_dim, classes):
        super().__init__()
        self.feat_concat = FeatureConcat()


        self.classifier_head = nn.Sequential(
            nn.Dropout(0.333),
            nn.Conv2d(in_channels=feature_dim*2,
                      out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_logits = nn.Sequential(
            nn.Dropout(0.333),
            nn.Linear(1024, classes)
        )

    def forward(self, feature_ori, feature_aug):
        # x和attentionparts_sameimg concat
        #   DONE (zihao): discussion on CONCATENATE or AVERAGE
        # feature_merge = torch.cat([feature_ori, feature_aug], dim=0)
        feature_merge = self.feat_concat(feature_ori, feature_aug)
        out = self.classifier_head(feature_merge)
        out = self.avgpool(out)
        # 拉直
        out = out.flatten(1)
        logits_cls = self.fc_logits(out)
        return logits_cls

class PartDiscriminator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feat_concat = FeatureConcat()

        self.classifier_head = nn.Sequential(
            nn.Dropout(0.333),
            nn.Conv2d(in_channels=feature_dim*2,
                      out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_discr = nn.Sequential(
            nn.Dropout(0.333),
            nn.Linear(1024, 2)
        )

    def forward(self, feature_ori, feature_aug):
        # x和attentionparts_sameimg concat
        #   DONE (zihao): discussion on CONCATENATE or AVERAGE
        # feature_merge = torch.cat([feature_ori, feature_aug], dim=0)
        feature_merge = self.feat_concat(feature_ori, feature_aug)
        out = self.classifier_head(feature_merge)
        out = self.avgpool(out)
        # 拉直
        out = out.flatten(1)
        # same img的dis结果
        logits_discr = self.fc_discr(out)

        return logits_discr


class AttentiveContrativeNetwork(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        #   Network Components
        #   ==  Un-parameterized
        self.feat_fusion = FeatureFusion(fusion_prob=param.fusion_prob)
        self.atten_augment = AttentionAugmentation(param.attention_num)
        self.atten_regularizer = AttentionRegularizer()
        #   ==  Parameterized
        self.backbone_global, self.feature_channels = REGISTRY['BACKBONE'][self.param.backbone](pretrained=True)
        self.backbone_local, self.feature_channels = REGISTRY['BACKBONE'][self.param.backbone](pretrained=True)
        self.attention_generator = AttentionGenerator(self.feature_channels, param.attention_num)
        self.classifier = PartClassifier(self.feature_channels, param.class_num)
        #   Xavier Initialization
        self.attention_generator = xavier_init(self.attention_generator)
        self.classifier = xavier_init(self.classifier)


    def forward(self, image=None, feature_ori=None, feature_aug_ori=None, augmented=None, mode='all'):
        if image is None:
            assert feature_ori is not None
        batch_size = image.size()[0] if image is not None else feature_ori.size()[0]
        if feature_ori is None:
            #   Get the original feature and attention
            feature_ori = self.backbone_global(image)
        if augmented is None:
            attention = self.attention_generator(feature_ori)
            augmented = self.atten_augment(image, attention, tolist=False)
        flattened = augmented.flatten(start_dim=0, end_dim=1)
        feature_aug = self.backbone_local(flattened)
        if feature_aug_ori is not None:
            feature_aug = self.feat_fusion(feature_aug_ori, feature_aug)
        feature_aug = feature_aug.view(
            batch_size, self.param.attention_num, *feature_ori.size()[1:])
        #   feature_ori with shape (batch_size, channel, f_h, f_w)
        #   feature_aug with shape (batch_size, num_attention, channel, f_h, f_w)
        logits_cls = self.classifier(feature_ori, feature_aug)
        if mode == 'feature_aug':
            return feature_aug
        elif mode == 'all':
            return feature_ori, augmented, feature_aug, logits_cls


@REGISTER_MODEL('AttentiveContrative')
class AttentiveContrativeModel():
    """
    Adversarial Attention Learning
    Please refer to APINet for interface reference
    The following code is based on the baseline implemnetation in this framework
    """

    def __init__(self, param, device=None, local_rank=0):
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.local_rank = local_rank
        '''
        self.seed = np.random.randint(low=0, high=2**32)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        '''
        self.param = param
        self.best_prec = 0
        #   Network Components
        self.adversarial_attention = AttentiveContrativeNetwork(self.param)
        self.atten_experience = AttentionExperience(param.class_num, self.adversarial_attention.feature_channels,
                                                    buffer_size=param.experience_num, seed=None)
        self.discriminator = PartDiscriminator(self.adversarial_attention.feature_channels)

        #   Xavier Initialization
        self.discriminator = xavier_init(self.discriminator)

        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = LSR(e=0.1)

        self.adversarial_attention.to(device=self.device)
        self.discriminator.to(device=self.device)
        self.criterion.to(device=self.device)

        # Bias in layers doesn't need decays
        self.optimizer_discriminator = optim.SGD(split_weights(self.discriminator),
                                                  self.param.lr, momentum=0.9, weight_decay=5e-4)
        self.optimizer_adatt = optim.SGD(split_weights(self.adversarial_attention),
                                              self.param.lr, momentum=0.9, weight_decay=5e-4)

        self.wrap_ddp()

        self.optimizers = [self.optimizer_adatt, self.optimizer_discriminator]

        self.module_dict = {'AttentiveContrativeNetwork': self.adversarial_attention,
                            'PartDiscriminator': self.discriminator,
                            'optimizer_adatt': self.optimizer_adatt,
                            'optimizer_discriminator': self.optimizer_discriminator
                            }

    def wrap_ddp(self):
        self.adversarial_attention = nn.parallel.DistributedDataParallel(self.adversarial_attention,
                                                                         device_ids=[self.local_rank],
                                                                         output_device=self.local_rank,
                                                                         find_unused_parameters=True)
        self.discriminator = nn.parallel.DistributedDataParallel(self.discriminator,
                                                                 device_ids=[self.local_rank],
                                                                 output_device=self.local_rank,
                                                                 find_unused_parameters=True)

    def train(self, input_var, target_var, image_ids=None, epoch=0, _iter=0, dataset=None):
        enable_neg = self.param.neg_learn and epoch >= 0

        self.adversarial_attention.train()
        self.discriminator.train()
        self.optimizer_adatt.zero_grad()
        batch_size = input_var.size()[0]

        #   Positive Learning and Classification
        feature_ori, augmented, feature_aug, logits_cls = self.adversarial_attention(input_var)
        loss_cls = self.criterion(logits_cls, target_var)

        loss_cls.backward()
        self.optimizer_adatt.step()

        feature_ori = feature_ori.detach()
        feature_aug = feature_aug.detach()

        loss_disc_pos = None
        self.optimizer_discriminator.zero_grad()
        if enable_neg:
            logits_disc = self.discriminator(feature_ori, feature_aug)
            loss_disc_pos = self.criterion(logits_disc,
                                           torch.ones(batch_size, dtype=torch.int64).to(device=self.device))
            self.atten_experience.add_experience(target_var.detach().cpu().numpy(),
                                                 augmented.detach().cpu().numpy(),
                                                 feature_aug.mean(1).detach().cpu())
            loss_disc_pos.backward()
        self.optimizer_discriminator.step()

        loss_cls = loss_cls.item()
        loss_disc_pos = loss_disc_pos.item() if loss_disc_pos is not None else 0
        feature_ori = feature_ori.detach()

        del image_ids, feature_aug

        loss_disc_neg = None
        self.optimizer_discriminator.zero_grad()
        self.optimizer_adatt.zero_grad()
        if enable_neg:
            #   Negative Learning
            with torch.no_grad():
                self.atten_experience.sort()
                list_augmented = []
                for n in range(batch_size):
                    _, augmented = \
                        self.atten_experience.sample(avoid_class=[target_var[n].cpu().numpy()],
                                                     num_experiences=1)
                    if augmented is not None:
                        list_augmented.extend(augmented)
            if len(list_augmented) > 0 and len(list_augmented) <= batch_size:
                batch_size = min(batch_size, len(list_augmented))
                augmented_t = torch.stack(list_augmented[:batch_size], dim=0)
                #   feature_aug should be (batch_size, exp_num, *feature_size)
                #   logits should (batch_size x class_num)
                augmented_t = augmented_t.to(self.device)
                feature_aug =\
                    self.adversarial_attention(augmented=augmented_t,
                                               feature_ori=feature_ori[:batch_size],
                                               mode='feature_aug')
                logits_disc = self.discriminator(feature_ori[:batch_size], feature_aug)
                loss_disc_neg = self.criterion(logits_disc,
                                               torch.zeros(batch_size, dtype=torch.int64).to(device=self.device))
                del list_augmented, augmented
                del augmented_t
            else:
                #   Fake loss for distributed data parallel
                batch_size = 1
                feature_aug =\
                    self.adversarial_attention(augmented=torch.randn(batch_size, self.param.attention_num, 3, 448, 448),
                                               feature_ori=feature_ori[:batch_size],
                                               mode='feature_aug')
                logits_disc = self.discriminator(feature_ori[:batch_size], feature_aug)
                loss_disc_neg = torch.mean(logits_disc - logits_disc)
            loss_disc_neg.backward()
        self.optimizer_adatt.step()
        self.optimizer_discriminator.step()
        loss_disc_neg = loss_disc_neg.item() if loss_disc_neg is not None else 0

        #   TODO (zihao): regulairzation need more discussion
        """
        ###################################regularizer改进
        coordinate, center_coordinate = self.atten_regularizer(attention_ori, feature_ori) ###这里应该是attention_ori吧？
        loss_regularizer = self.criterion2(coordinate, center_coordinate)
        ##############################################
        """

        return loss_disc_neg+loss_disc_pos+loss_cls, logits_cls, target_var

    def evaluate(self, input_var, target_var):
        with torch.no_grad():
            self.adversarial_attention.eval()
            # compute output
            #   Upper branch
            _, _, _, logits_cls =\
                self.adversarial_attention(input_var)
            loss = self.criterion(logits_cls, target_var)
        return loss, logits_cls

    def state_dict(self):
        state_dict = {}
        for module in self.module_dict.keys():
            state_dict[module] = self.module_dict[module].state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for module in self.module_dict.keys():
            self.module_dict[module].load_state_dict(state_dict[module])

    def to(self, device=None):
        self.adversarial_attention.to(device=device)
        self.discriminator.to(device=device)
        self.criterion.to(device=device)
        return self



if __name__ == '__main__':
    def build_exp(exp, classes, buff_size):
        #   Build Experience
        for n in range(classes):
            exp.add_experience([n] * buff_size,
                               np.random.rand(buff_size, 3, 448, 448),
                               np.random.rand(buff_size, 512, 14, 14))
        return exp


    from fgvc_cl.utils.ConfigManager import ConfigManager
    print_loaded(REGISTRY)
    c = ConfigManager()
    c.load('../../config/AttentiveContrative.yaml')
    c.attention_num = 2
    c.batch_size = 2
    c.class_num = 200
    model = AttentiveContrativeModel(c)
    model.load_state_dict(model.state_dict())
    print("load & saved")

    model.atten_experience = build_exp(
        model.atten_experience, c.class_num, c.attention_num*4)
    random_label = torch.randint(c.class_num, size=(c.batch_size,))
    model.train(torch.rand(2, 3, 448, 448), random_label, image_ids=['0', '0'])

