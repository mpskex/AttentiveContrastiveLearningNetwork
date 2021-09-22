"""
Training script for Pollen Challenge
Fangrui Liu @ UBCO ISDPRL

Copyright Reserved 2020
NO DISTRIBUTION AGREEMENT PROVIDED
"""

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

class ConfigNode():
    pass

class ConfigManager():
    def __init__(self):
        self.lr = 1e-3
        self.lr_decay_rate = 0
        self.lr_decay_epochs = 0
        self.epochs = 100
        self.batch_size = 128
        self.data_root = "dataset/"
        self.data_loader = "FGVC_Aircraft"
        self.trainer = 'APINetTrainer'
        self.backbone = "resnet50"
        self.model = "AttentiveContrative"
        self.attention_num = 16
        self.augmentation = "Normal"
        self.class_num = 100
        self.experience_num = 64
        self.name = None
        self.mean_pixel = None
        self.range = 255
        self.pretrained = False
        self.std_pixel = None

    def __to_prop__(self, param, root=None):
        if root is None:
            root = self
        for p in param.keys():
            if isinstance(param[p], dict):
                root.__setattr__(p, ConfigNode())
                self.__to_prop__(param[p], root.__getattribute__(p))
            else:
                root.__setattr__(p, param[p])

    def load(self, filename):
        with open(filename, 'r') as f:
            param = yaml.load(f.read(), Loader=Loader)
            f.close()
        self.__to_prop__(param)
        print('Loaded %s!' % (filename))

    def __str__(self, root=None, depth=None):
        _str = ""
        if root is None:
            root = self
        for p in dir(root):
            attr = root.__getattribute__(p)
            if not callable(attr) and '__' not in p:
                if isinstance(attr, ConfigNode):
                    _str += self.__str__(root=attr, depth=depth + '.' + p if depth is not None else p)
                else:
                    _str += "%20s :\t %s\r\n" % (depth + '.' + p if depth is not None else p, str(attr))
        return _str

    def write(self, filename):
        with open(filename, 'w') as f:
            f.write(yaml.dump(self.__dict__, Dumper=Dumper))
            f.close()



if __name__ == '__main__':
    c = ConfigManager()
    c.write('../../config/Default.yaml')
    c.load('../../config/Default.yaml')
    c.test = {
        'a': {'c': 2},
        'b': 1
    }
    c.batch_size = 64
    print("to str")
    print(str(c))

    
