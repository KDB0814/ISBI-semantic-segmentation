import re
import os
import cv2
import torch

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
        self.img_path = f'../../Data/ISBI/{self.mode}/image/aug'
        self.lbl_path = f'../../Data/ISBI/{self.mode}/label/aug'
        self.img_list = sorted(os.listdir(self.img_path), key=sorter)
        self.lbl_list = sorted(os.listdir(self.lbl_path), key=sorter)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_path, self.img_list[idx]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(os.path.join(self.lbl_path, self.lbl_list[idx]), cv2.IMREAD_GRAYSCALE)

        # Make numpy to tensor
        # H W C => C H W
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        lbl = torch.from_numpy(lbl)
        lbl = lbl.unsqueeze(0)

        sample = {'image': img, 'label': lbl}

        if self.transform:
            sample = self.transform(sample)

        return sample


