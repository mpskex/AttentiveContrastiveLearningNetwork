"""
PyTorch dataset for CUB-200-2011
https://github.com/TDeVries/cub2011_dataset
"""

import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from fgvc_cl.data.Transforms import get_transform
from fgvc_cl.utils.Register import REGISTER_DATALOADER

class CUBDataset(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transform(train)
        self.loader = default_loader
        self.train = train
        self.class_num = 200

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)
        os.remove(os.path.join(self.root, self.filename))

    def __len__(self):
        return len(self.data)

    def _get_label(self, idx):
        return self.data.iloc[idx].target - 1

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


@REGISTER_DATALOADER('CUB_200_2011')
def build_CUB_200_2011(param, split='train', tta=False):
    import inspect
    from os.path import dirname, join
    root = join(dirname(dirname(dirname(inspect.getabsfile(CUBDataset)))), param.data_root)
    if split == 'train':
        _train = True
        transform = get_transform(True, tta, rot=30)
    elif split == 'valid':
        _train = False
        transform = get_transform(False, tta, rot=30)
    else:
        _train = True
        transform = get_transform(True, tta, rot=30)
    download = True
    return CUBDataset(root=root, train=_train, download=download, transform=transform)

if __name__ == '__main__':
    class cfg():
        def __init__(self):
            self.data_root = 'dataset/'

    dataset = build_CUB_200_2011(cfg(), split='train')
    image, label = dataset[0]
    print(type(image), type(label))