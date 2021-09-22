import torch
import inspect
from os.path import dirname, join
from os import makedirs

def save_checkpoint(name, param, filename, state, is_best):
    root = join(dirname(dirname(dirname(inspect.getabsfile(save_checkpoint)))))
    makedirs(join(root, 'models', name), exist_ok=True)
    #torch.save(state, join(root, 'models', name, filename))
    param.write(join(root, 'models', name, 'config.yaml'))
    if is_best:
        torch.save(state, join(root, 'models', name, 'model_best.pth.tar'))

def load_checkpoint(name, filename=None, is_best=True):
    root = join(dirname(dirname(dirname(inspect.getabsfile(save_checkpoint)))))
    if is_best or filename is None:
        filename = 'model_best.pth.tar'
    d = torch.load(join(root, 'models', name, filename))
    return d['epoch'], d['best_prec'], d['state_dict']

