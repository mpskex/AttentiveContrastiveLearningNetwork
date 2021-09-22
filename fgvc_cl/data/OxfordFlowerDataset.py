"""
Stanford-dogs/data/oxford_flowers.py
https://github.com/zrsmithson/Stanford-dogs/blob/master/data/oxford_flowers.py
"""
from __future__ import print_function

from PIL import Image
from os.path import join
import os
import scipy.io
import numpy as np

import torch.utils.data as data
from torchvision.datasets.utils import download_url, list_dir, list_files

from fgvc_cl.data.Transforms import get_transform
from fgvc_cl.utils.Register import REGISTER_DATALOADER


class flowers(data.Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'OxfordFlowers'
    download_url_prefix = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102'

    def __init__(self,
                 root,
                 train=True,
                 val=False,
                 transform=None,
                 target_transform=None,
                 download=False,
                 classes=None):

        self.root = join(os.path.expanduser(root), self.folder)
        self.train = train
        self.val = val
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transform(train or not val)
        self.target_transform = target_transform
        self.class_num = 102

        if download:
            self.download()

        self.split = self.load_split()
        # self.split = self.split[:100]  # TODO: debug only get first ten classes

        self.images_folder = join(self.root, 'jpg')

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image_name, target_class = self.split[index]
        image_path = join(self.images_folder, "image_%05d.jpg" % (image_name+1))
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class.astype(np.int)

    def _get_label(self, idx):
        _, target_class = self.split[idx]
        if self.target_transform:
            target_class = self.target_transform(target_class)
        return target_class

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'jpg')) and os.path.exists(join(self.root, 'imagelabels.mat')) and os.path.exists(join(self.root, 'setid.mat')):
            if len(os.listdir(join(self.root, 'jpg'))) == 8189:
                print('Files already downloaded and verified')
                return

        filename = '102flowers'
        tar_filename = filename + '.tgz'
        url = self.download_url_prefix + '/' + tar_filename
        download_url(url, self.root, tar_filename, None)
        with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
            tar_file.extractall(self.root)
        os.remove(join(self.root, tar_filename))

        filename = 'imagelabels.mat'
        url = self.download_url_prefix + '/' + filename
        download_url(url, self.root, filename, None)

        filename = 'setid.mat'
        url = self.download_url_prefix + '/' + filename
        download_url(url, self.root, filename, None)

    def load_split(self):
        split = scipy.io.loadmat(join(self.root, 'setid.mat'))
        labels = scipy.io.loadmat(join(self.root, 'imagelabels.mat'))['labels']
        if self.train:
            split = split['trnid']
        elif self.val:
            split = split['valid']
        else:
            split = split['tstid']

        split = list(split[0] - 1) # set it all back 1 as img indexs start at 1
        labels = list(labels[0][split]-1)
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self.split)):
            image_name, target_class = self.split[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)"%(len(self.split), len(counts.keys()), float(len(self.split))/float(len(counts.keys()))))

        return counts

@REGISTER_DATALOADER('Oxford_Flower')
def build_Oxford_flowers(param, split='train', tta=False):
    import inspect
    from os.path import dirname, join
    root = join(dirname(dirname(dirname(inspect.getabsfile(flowers)))), param.data_root)
    if split == 'train':
        _train = True
        _valid = False
        transform = get_transform(True, tta, rot=45)
    elif split == 'valid':
        _train = False
        _valid = True
        transform = get_transform(False, tta, rot=45)
    elif split == 'test':
        _train = False
        _valid = False
        transform = get_transform(False, tta, rot=45)
    else:
        _train = True
        _valid = False
        transform = get_transform(True, tta, rot=45)
    return flowers(root=root, train=_train, val=_valid, download=True, transform=transform)

if __name__ == '__main__':
    class cfg():
        def __init__(self):
            self.data_root = 'dataset/'

    dataset = build_Oxford_flowers(cfg(), split='train')
    image, label = dataset[0]
    print(type(image), type(label))