"""
Add FGVC-Aircraft Dataset
https://github.com/pytorch/vision/pull/467/files
"""

from __future__ import print_function
import torch.utils.data as data
from torchvision.datasets.folder import pil_loader, accimage_loader, default_loader
from torchvision.datasets.utils import download_url
import os
import numpy as np

from fgvc_cl.data.Transforms import get_transform
from fgvc_cl.utils.Register import REGISTER_DATALOADER


def make_dataset(dir, image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'fgvc-aircraft-2013b', 'data', 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images


def find_classes(classes_file):
    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split('\n')[0].split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)


class FGVCAircraft(data.Dataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft>`_ Dataset.

    Args:
        root (string): Root directory path to dataset.
        class_type (string, optional): The level of FGVC-Aircraft fine-grain classification
            to label data with (i.e., ``variant``, ``family``, or ``manufacturer``).
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(self, root, class_type='variant', split='train', transform=None,
                 target_transform=None, loader=default_loader, download=False):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))
        self.root = os.path.expanduser(root)
        self.class_type = class_type
        self.split = split
        self.class_num = 100
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = find_classes(self.classes_file)
        samples = make_dataset(self.root, image_ids, targets)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transform(train='train' in split)
        self.target_transform = target_transform
        self.loader = loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    
    def _get_label(self, idx):
        _, target = self.samples[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return target

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'fgvc-aircraft-2013b', 'data')) and \
            os.path.exists(self.classes_file)

    def download(self):
        """Download the FGVC-Aircraft data if it doesn't exist already."""
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s ... (may take a few minutes)' % self.url)
        tar_name = self.url.rpartition('/')[-1]
        tar_path = os.path.join(self.root, tar_name)

        # download .tar.gz file
        download_url(self.url, self.root, tar_name)

        # extract .tar.gz to PARENT_DIR/fgvc-aircraft-2013b
        data_folder = tar_path.strip('.tar.gz')
        print('Extracting %s to %s ... (may take a few minutes)' % (tar_path, data_folder))
        tar = tarfile.open(tar_path)
        tar.extractall(self.root)

        # delete .tar.gz file
        print('Deleting %s ...' % tar_path)
        os.remove(tar_path)

        print('Done!')

@REGISTER_DATALOADER('FGVC_Aircraft')
def build_FGVCAircraftDataset(param, split='train', tta=False):
    import inspect
    from os.path import dirname, join
    root = join(dirname(dirname(dirname(inspect.getabsfile(FGVCAircraft)))), param.data_root)
    class_type = 'variant'
    download = True
    if split == 'train':
        _split = 'train'
        transform = get_transform(True, tta, rot=30)
    elif split == 'valid':
        _split = 'val'
        transform = get_transform(False, tta, rot=30)
    else:
        _split = 'train'
        transform = get_transform(True, tta, rot=30)
    return FGVCAircraft(root=root, split=_split, class_type=class_type, download=download, transform=transform)



if __name__ == '__main__':
    class cfg():
        def __init__(self):
            self.data_root = 'dataset/'

    dataset = build_FGVCAircraftDataset(cfg(), split='train')
    image, label = dataset[0]
    print(type(image), type(label))