import torchvision.transforms as T
import random
import torch
from torchvision.transforms import InterpolationMode

def apply_mirroring(image_tensor, mirroring_size=94):
    # Pad image tensor
    padded_image = torch.nn.functional.pad(image_tensor, (mirroring_size, mirroring_size, mirroring_size, mirroring_size), mode='reflect')

    return padded_image

class Random_processing(object):
    def __init__(self):
        self.h_flip = T.RandomHorizontalFlip()
        self.shift = T.RandomAffine(translate=(.1, .1), degrees=0)
        self.gray_value = T.ColorJitter(brightness=0.3 + 0.1*random.random())
        self.elastic = T.ElasticTransform(interpolation=InterpolationMode.NEAREST)
        self.normalization = T.Normalize(mean=[.491], std=[.1662])


    def __call__(self, samples):
        img, lbl= samples['image'], samples['label']

        # Normalization

        if random.random() > 0.5:
            img, lbl = self.shift(img), self.shift(lbl)

        if random.random() > 0.5:
            img, lbl = self.h_flip(img), self.shift(lbl)

        if random.random() > 0.5:
            img = self.gray_value(img)

        if random.random() > 0.5:
            img, lbl = self.elastic(img), self.elastic(lbl)
        origin = img
        img = apply_mirroring(img)
        img = self.normalization(img)

        lbl = lbl.to(torch.int)
        sample = {'image': img, 'label': lbl, 'origin' : origin}

        return sample

class Inference_processing(object):
    def __init__(self):
        self.normalization = T.Normalize(mean=[.491], std=[.1662])

    def __call__(self, samples):
        img, lbl, ori = samples['image'], samples['label'], samples['origin']

        img = apply_mirroring(img)
        img = self.normalization(img)

        lbl = lbl.to(torch.int)
        sample = {'image': img, 'label': lbl, 'origin' : ori}

        return sample
