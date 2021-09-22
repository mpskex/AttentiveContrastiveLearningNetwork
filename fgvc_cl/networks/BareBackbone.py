import torch
from torch import nn
from torch.nn import functional as F

from fgvc_cl.utils.Register import REGISTER_MODEL, REGISTRY
from fgvc_cl.utils.Tricks import split_weights

"""
By Fangrui Liu (fangrui.liu@ubc.ca) Zihao Liu(lzh19961031@pku.edu.cn)
University of British Columbia Okanagan
Peking University
"""


class BareBackbone(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        self.backbone, feature_channels = REGISTRY['BACKBONE'][self.param.backbone](
            pretrained=True)

        self.fc = nn.Sequential(
            nn.Linear(feature_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(512, self.param.class_num)
        )

    def forward(self, input):
        feature = self.backbone(input)
        return self.fc(F.adaptive_avg_pool2d(feature, 1).squeeze())


@REGISTER_MODEL("BareBackbone")
class BareBackboneModel():
    def __init__(self, param, device=None, local_rank=0):
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.param = param
        self.local_rank = local_rank
        self.best_prec = 0
        self.model = BareBackbone(self.param)
        self.model = self.model.to(device=self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(split_weights(
            self.model), self.param.lr, momentum=0.9, weight_decay=5e-4)
        self.optimizers = [self.optimizer]
        self.module_dict = {'state_dict': self.model,
                            'optimizer': self.optimizer
                            }
        self.wrap_ddp()

    def wrap_ddp(self):
        self.model = nn.parallel.DistributedDataParallel(self.model,
                                                         device_ids=[self.local_rank],
                                                         output_device=self.local_rank,
                                                         find_unused_parameters=True)

    def train(self, input_var, target_var, image_ids=None, epoch=0, _iter=0, dataset=None):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(input_var)
        loss = self.criterion(logits, target_var)
        loss.backward()
        self.optimizer.step()
        return loss.item(), logits, target_var

    def evaluate(self, input_var, target_var):
        with torch.no_grad():
            self.model.eval()
            logits = self.model(input_var)
            loss = self.criterion(logits, target_var)
        return loss.item(), logits

    def inference(self, input_var):
        self.model.eval()
        return self.model(input_var)

    def state_dict(self):
        state_dict = {}
        for module in self.module_dict.keys():
            state_dict[module] = self.module_dict[module].state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for module in self.module_dict.keys():
            self.module_dict[module].load_state_dict(state_dict[module])

    def to(self, device=None):
        self.model.to(device=device)
        return self
