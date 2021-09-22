"""
Fangrui Liu @ UBCO ISDPRL

Copyright Reserved 2020
NO DISTRIBUTION AGREEMENT PROVIDED
"""


from torch import nn
from torchvision import models
from fgvc_cl.utils import Register
from efficientnet_pytorch import EfficientNet
from resnest.torch import resnest50, resnest101, resnest200

def build_linear(features, classes):
    return nn.Linear(features, classes)


@Register.REGISTER_BACKBONE('resnet18')
def build_resnet18(pretrained=False, device=None):
    """
    Build resnet18
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    model = models.resnet18(pretrained=pretrained).to(device=device)
    layers = list(model.children())[:-2]
    conv = nn.Sequential(*layers)
    return conv, 512


@Register.REGISTER_BACKBONE('resnet34')
def build_resnet34(pretrained=False, device=None):
    """
    Build resnet34
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    model = models.resnet34(pretrained=pretrained).to(device=device)
    layers = list(model.children())[:-2]
    conv = nn.Sequential(*layers)
    return conv, 512


@Register.REGISTER_BACKBONE('resnet50')
def build_resnet50(pretrained=False, device=None):
    """
    Build resnet50
    :param classes:
    :param pretrained:
    :param device:
    :return conv:
    :return feature_channel:
    """
    model = models.resnet50(pretrained=pretrained).to(device=device)
    layers = list(model.children())[:-2]
    conv = nn.Sequential(*layers)
    return conv, 2048


@Register.REGISTER_BACKBONE('resnet101')
def build_resnet101(pretrained=False, device=None):
    """
    Build resnet50
    :param classes:
    :param pretrained:
    :param device:
    :return conv:
    :return feature_channel:
    """
    model = models.resnet101(pretrained=pretrained).to(device=device)
    layers = list(model.children())[:-2]
    conv = nn.Sequential(*layers)
    return conv, 2048

@Register.REGISTER_BACKBONE('resnet152')
def build_resnet152(pretrained=False, device=None):
    """
    Build resnet50
    :param classes:
    :param pretrained:
    :param device:
    :return conv:
    :return feature_channel:
    """
    model = models.resnet152(pretrained=pretrained).to(device=device)
    layers = list(model.children())[:-2]
    conv = nn.Sequential(*layers)
    return conv, 2048


@Register.REGISTER_BACKBONE('resnext101')
def build_resnext101(pretrained=False, device=None):
    """
    Build resnext101
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    model = models.resnext101_32x8d(pretrained=pretrained).to(device=device)
    layers = list(model.children())[:-2]
    conv = nn.Sequential(*layers)
    return conv, 2048


@Register.REGISTER_BACKBONE('resnext50')
def build_resnext50(pretrained=False, device=None):
    """
    Build resnext50
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    model = models.resnext50_32x4d(pretrained=pretrained).to(device=device)
    layers = list(model.children())[:-2]
    conv = nn.Sequential(*layers)
    return conv, 2048


@Register.REGISTER_BACKBONE('vgg11')
def build_vgg11(pretrained=False, device=None):
    """
    Build vgg11
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    model = models.vgg11_bn(pretrained=pretrained).to(device=device)
    layers = list(model.children())[:-2]
    conv = nn.Sequential(*layers)
    return conv, 512


@Register.REGISTER_BACKBONE('vgg13')
def build_vgg13(pretrained=False, device=None):
    """
    Build vgg13
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    model = models.vgg13_bn(pretrained=pretrained).to(device=device)
    layers = list(model.children())[:-2]
    conv = nn.Sequential(*layers)
    return conv, 512


@Register.REGISTER_BACKBONE('vgg16')
def build_vgg16(pretrained=False, device=None):
    """
    Build vgg16
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    model = models.vgg16_bn(pretrained=pretrained).to(device=device)
    layers = list(model.children())[:-2]
    conv = nn.Sequential(*layers)
    return conv, 512


@Register.REGISTER_BACKBONE('vgg19')
def build_vgg19(pretrained=False, device=None):
    """
    Build vgg19
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    model = models.vgg19_bn(pretrained=pretrained).to(device=device)
    layers = list(model.children())[:-2]
    conv = nn.Sequential(*layers)
    return conv, 512


class EffcientFeatureExtractor(nn.Module):
    def __init__(self, efficientnet):
        super(EffcientFeatureExtractor, self).__init__()
        self.efficientnet = efficientnet
        del self.efficientnet._fc, self.efficientnet._avg_pooling, self.efficientnet._dropout

    def forward(self, inputs):
        return self.efficientnet.extract_features(inputs)



@Register.REGISTER_BACKBONE('efficientnet-b4')
def build_efficientnet_b4(pretrained=False, device=None):
    """
    Build efficientnet-b4
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b4').to(device=device)
    else:
        model = EfficientNet.from_name('efficientnet-b4').to(device=device)
    return EffcientFeatureExtractor(model), 1792

@Register.REGISTER_BACKBONE('efficientnet-b5')
def build_efficientnet_b5(pretrained=False, device=None):
    """
    Build efficientnet-b5
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b5').to(device=device)
    else:
        model = EfficientNet.from_name('efficientnet-b5').to(device=device)
    return EffcientFeatureExtractor(model), 2048

@Register.REGISTER_BACKBONE('efficientnet-b6')
def build_efficientnet_b6(pretrained=False, device=None):
    """
    Build efficientnet-b6
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b6').to(device=device)
    else:
        model = EfficientNet.from_name('efficientnet-b6').to(device=device)
    return EffcientFeatureExtractor(model), 2304


@Register.REGISTER_BACKBONE('efficientnet-b2')
def build_efficientnet_b2(pretrained=False, device=None):
    """
    Build efficientnet-b2
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b2').to(device=device)
    else:
        model = EfficientNet.from_name('efficientnet-b2').to(device=device)
    return EffcientFeatureExtractor(model), 1408

@Register.REGISTER_BACKBONE('resnest50')
def build_resnest50(pretrained=False, device=None):
    """
    Build resnest50
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    model = resnest50(pretrained=pretrained).to(device=device)
    layers = list(model.children())[:-2]
    conv = nn.Sequential(*layers)
    return conv, 2048

@Register.REGISTER_BACKBONE('resnest101')
def build_resnest101(pretrained=False, device=None):
    """
    Build resnest101
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    model = resnest101(pretrained=pretrained).to(device=device)
    layers = list(model.children())[:-2]
    conv = nn.Sequential(*layers)
    return conv, 2048


@Register.REGISTER_BACKBONE('resnest200')
def build_resnest200(pretrained=False, device=None):
    """
    Build resnest200
    :param classes:
    :param pretrained:
    :param device:
    :return:
    """
    model = resnest200(pretrained=pretrained).to(device=device)
    layers = list(model.children())[:-2]
    conv = nn.Sequential(*layers)
    return conv, 2048

@Register.REGISTER_BACKBONE('densenet161')
def build_densenet161(pretrained=False, device=None):
    model = nn.Sequential(models.densenet161(pretrained=pretrained).features,
                          nn.ReLU(inplace=True)).to(device=device)
    return model, 2208


if __name__ == '__main__':
    """
    Unit test for the backbone
    """
    print(Register.REGISTRY)
    import numpy as np
    from torch import as_tensor
    from torchsummary import summary

    for name in ['resnet101', 'densenet161']:
        model, channel = Register.REGISTRY['BACKBONE'][name](pretrained=True)
        out = model(as_tensor(np.random.rand(1, 3, 64, 64).astype(np.float32)))
        summary(model, (3, 448, 448))
        print("{}'s feature channel is {}".format(name, out.size(1)))
        print("Building function for  %s tested!" % (name))