from torch.utils.data import Dataset


"""
Image Index Dataset Wrapper

by Fangrui Liu (fangrui.liu@ubc.ca)
Copyright reserved. 2020
"""

class ImageIndexDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_num = dataset.class_num

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return image, target, index

    def __len__(self):
        return len(self.dataset)
    
    def get_label(self, idx):
        return self.dataset._get_label(idx)