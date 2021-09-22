import os
import scipy
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets.utils import download_url

from fgvc_cl.data.Transforms import get_transform
from fgvc_cl.utils.Register import REGISTER_DATALOADER


class StanfordCarsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, split='train', cleaned=None, transform=None, download=True):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.urls = {"train": "http://imagenet.stanford.edu/internal/car196/cars_train.tgz",
                     "test": "http://imagenet.stanford.edu/internal/car196/cars_test.tgz",
                     "meta": "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz", }

        self.md5 = {
            "train":"",
            "test":"",
            "meta":"",
            }

        self.root = os.path.join(root, 'StanfordCars')
        os.makedirs(self.root, exist_ok=True)
        if download:
            self.download()

        mat_anno = os.path.join(self.root, 'devkit', 'cars_'+split+'_annos.mat')
        car_names = os.path.join(self.root, 'devkit', 'cars_meta.mat')
        data_dir = os.path.join(self.root, 'cars_'+split)

        self.full_data_set = scipy.io.loadmat(mat_anno)
        self.car_annotations = self.full_data_set['annotations']
        self.car_annotations = self.car_annotations[0]

        if cleaned is not None:
            cleaned_annos = []
            print("Cleaning up data set (only take pics with rgb chans)...")
            clean_files = np.loadtxt(cleaned, dtype=str)
            for c in self.car_annotations:
                if c[-1][0] in clean_files:
                    cleaned_annos.append(c)
            self.car_annotations = cleaned_annos

        self.car_names = scipy.io.loadmat(car_names)['class_names']
        self.car_names = np.array(self.car_names[0])

        self.classes = self.car_names.shape[0]
        self.class_num = self.classes

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.data_dir, self.car_annotations[idx][-1][0])
        image = Image.open(img_name)
        car_class = self.car_annotations[idx][-2][0][0]

        if self.transform:
            image = self.transform(image)

        return image, torch.as_tensor(car_class.astype(np.int64), dtype=torch.long)

    def _check_exist(self):
        return os.path.exists(os.path.join(self.root, 'data')) and \
            os.path.exists(self.classes_file)

    def download(self):
        import tarfile
        for n in self.urls.keys():
            if not os.path.exists(os.path.join(self.root, 'cars_'+n)):
                download_url(self.urls[n], self.root, n+'.tgz')
                with tarfile.open(os.path.join(self.root, n+'.tgz'), "r:gz") as tar:
                    tar.extractall(path=self.root)
                # os.remove(os.path.join(self.root, n+'.tgz'))
            else:
                print("File already downloaded and parsed.")


    def map_class(self, id):
        id = np.ravel(id)
        ret = self.car_names[id - 1][0][0]
        return ret


@REGISTER_DATALOADER('Stanford_Cars')
def build_Stanford_Cars(param, split='train', tta=False):
    import fgvc_cl
    import inspect
    from os.path import dirname, join
    root = join(dirname(dirname(inspect.getabsfile(fgvc_cl))), param.data_root)

    if split == 'train':
        split = 'train'
        train = True
    elif split == 'valid':
        split = 'train'
        train = False
    else:
        split = 'train'
        train = True
    return StanfordCarsDataset(root=root, split=split, transform=get_transform(train, tta, rot=15))

if __name__ == '__main__':
    class cfg():
        def __init__(self):
            self.data_root = 'dataset/'

    dataset = build_Stanford_Cars(cfg(), split='valid')
    image, label = dataset[-1]
    print(len(dataset), dataset.class_num)
    print(type(image), type(label))
    print(label)