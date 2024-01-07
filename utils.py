import torchvision.transforms as T
import random
import numpy as np
import torch
from torchvision.transforms import InterpolationMode

def apply_mirroring(image, mirroring_size=94):
    image = image.permute(1, 2, 0)
    mirrored_image = np.pad(image, pad_width=((mirroring_size, mirroring_size),
                                              (mirroring_size, mirroring_size),
                                              (0, 0)), mode='reflect')
    mirrored_image = mirrored_image.transpose(2, 0, 1)
    return mirrored_image


class Random_processing(object):
    def __init__(self):
        self.h_flip = T.RandomHorizontalFlip()
        self.shift = T.RandomAffine(translate=(.001, .001), degrees=0)
        self.gray_value = T.ColorJitter(brightness=0.2 + 0.1*random.random())
        self.elastic = T.ElasticTransform(interpolation=InterpolationMode.NEAREST)

    def __call__(self, samples):
        img, lbl = samples['image'], samples['label']

        if random.random() > 0.5:
            img, lbl = self.shift(img), self.shift(lbl)

        if random.random() > 0.5:
            img, lbl = self.h_flip(img), self.shift(lbl)

        if random.random() > 0.5:
            img = self.gray_value(img)

        if random.random() > 0.5:
            img, lbl = self.elastic(img), self.elastic(lbl)

        img = apply_mirroring(img)
        img = torch.from_numpy(img)

        img = img/255.0
        lbl = lbl/255.0

        sample = {'image': img, 'label': lbl}

        return sample
