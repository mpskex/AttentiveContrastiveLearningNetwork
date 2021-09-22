# encoding: utf-8
import torch
from PIL import ImageFilter
import random
import numpy as np
from torchvision import transforms
mean_std = {
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225)
}

"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import math
import random


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

def denorm(image):
    mean = np.expand_dims(np.expand_dims(mean_std['mean'], 1), 1)
    std = np.expand_dims(np.expand_dims(mean_std['std'], 1), 1)
    image = list(map(lambda x: x[0] * std + mean, np.split(image, image.shape[0])))
    return np.array(image, dtype=np.uint8)

def norm(image):
    mean = np.expand_dims(np.expand_dims(mean_std['mean'], 1), 1)
    std = np.expand_dims(np.expand_dims(mean_std['std'], 1), 1)
    image = list(map(lambda x: (x[0] - mean) / std, np.split(image, image.shape[0])))
    return np.array(image, dtype=np.float32)

def get_transform(train=True, tta=False, rot=45):
    if train:
        transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.Pad(20),
            transforms.RandomCrop([448, 448]),
            transforms.RandomRotation(rot),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                **mean_std
            ),
            # RandomErasing(probability=0.5, mean=mean_std['mean'])
        ])
        return transform
    else:
        if not tta:
            transform = transforms.Compose([
                transforms.Resize([512, 512]),
                transforms.CenterCrop([448, 448]),
                transforms.ToTensor(),
                transforms.Normalize(
                    **mean_std
                )])
            return transform
        else:
            transform = transforms.Compose([
                transforms.Resize([512, 512]),
                transforms.ToTensor(),
                transforms.Normalize(
                    **mean_std
                )])
            return transform