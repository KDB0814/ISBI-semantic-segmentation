import re
import os

from torchvision.io import read_image
from torch.utils.data import Dataset


def sorter(text):
    num = re.findall(r'\d+', text)
    return int(num[0]) if num else 0


class ISBI(Dataset):
    def __init__(self, mode, transform=None):
        '''
        ISBI 2012 dataset
            - mode : 'train' or 'valid'
            - transform : Data augmentation (default : None)
        '''
        self.mode = mode
        self.transform = transform
        self.img_path = f'../../Data/ISBI/prep/{self.mode}/image'
        self.lbl_path = f'../../Data/ISBI/prep/{self.mode}/label'
        self.img_list = sorted(os.listdir(self.img_path), key=sorter)
        self.lbl_list = sorted(os.listdir(self.lbl_path), key=sorter)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = read_image(os.path.join(self.img_path, self.img_list[idx]))/255.0
        lbl = read_image(os.path.join(self.lbl_path, self.lbl_list[idx]))/255.0

        sample = {'image': img, 'label': lbl, 'origin' : img}

        if self.transform:
            sample = self.transform(sample)

        return sample