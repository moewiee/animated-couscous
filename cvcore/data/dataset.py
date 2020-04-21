from collections import OrderedDict
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import gc
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations import pytorch
from .auto_augment import Invert, RandAugment, AugmentAndMix
import albumentations
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
import torchvision


def train_aug():
    return A.Compose([
        A.Rotate(10),
        A.HorizontalFlip(),
        A.pytorch.ToTensor()
    ], p=1)


def valid_aug():
    return A.Compose([
        A.pytorch.ToTensor()
    ], p=1)


def crop_and_resize_images(orig_images, resize_size):
    resized_images = []
    for img in tqdm(orig_images):
        img = img.reshape(ORIG_IMG_SIZE[:-1])
        _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        idx = 0
        ls_xmin = []
        ls_ymin = []
        ls_xmax = []
        ls_ymax = []
        for cnt in contours:
            idx += 1
            x,y,w,h = cv2.boundingRect(cnt)
            ls_xmin.append(x)
            ls_ymin.append(y)
            ls_xmax.append(x + w)
            ls_ymax.append(y + h)
        xmin = min(ls_xmin)
        ymin = min(ls_ymin)
        xmax = max(ls_xmax)
        ymax = max(ls_ymax)
        roi = img[ymin:ymax,xmin:xmax]
        resized_roi = cv2.resize(roi, (resize_size, resize_size)).reshape(1, -1)
        resized_images.append(resized_roi)
    resized_images = np.concatenate(resized_images, 0)
    return resized_images


def to_tensor(image):
    image = torch.from_numpy(np.asarray(image)).float()
    try:
        image = image.permute(2,0,1)
    except:
        print("image", image.size())
    image.div_(255.)
    return image


class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.

    Author: Qishen Ha
    Email: haqishen@gmail.com
    2020/01/29

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value

                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')


class WDataset(Dataset):
    def __init__(self, images, label, mode='train', cfg=None):
        self.images = images
        self.mode = mode
        assert self.mode in ("train", "valid", "test")
        self.dir = cfg.DIRS.TRAIN_IMAGES if mode != 'test' else cfg.DIRS.TEST_IMAGES
        self.size = (cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)
        if self.mode !="test":
            self.label = label
        if self.mode == "train":
            if cfg.DATA.AUGMENT == "randaug":
                self.transform = RandAugment(n=cfg.DATA.RANDAUG.N,
                    m=cfg.DATA.RANDAUG.M)
            elif cfg.DATA.AUGMENT == "augmix":
                self.transform = AugmentAndMix(k=cfg.DATA.RANDAUG.N,
                    m=cfg.DATA.RANDAUG.M)
        self.grid_mask = albumentations.Compose([
                                        albumentations.OneOf(
                                            (GridMask(num_grid=(3,7), rotate=15, mode=0),
                                             GridMask(num_grid=(3,7), rotate=15, mode=1),
                                             GridMask(num_grid=(3,7), rotate=15, mode=2)),
                                        p=1)
                                    ]) if cfg.DATA.GRIDMASK else None
        self.resize_crop = torchvision.transforms.Compose(
            [torchvision.transforms.RandomResizedCrop(
                self.size,scale=(0.7, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.mode != "test":
            lb = self.label[idx].astype(np.float32)
            lb = torch.Tensor([lb])
        image = Image.open(os.path.join(self.dir, self.images[idx] + ".jpg")).convert("RGB")
        image = image.resize(self.size)
        if self.mode == "train":
            if self.grid_mask:
                image = np.array(image)
                image = self.grid_mask(image=image)
                image = image['image']
                image = Image.fromarray(image)
            ### RandAugment ###
            if isinstance(self.transform, RandAugment):
                image = self.transform(image)
                image = to_tensor(image)
                return image, lb
            ### AugMix ###
            elif isinstance(self.transform, AugmentAndMix):
                image, aug1, aug2 = self.transform(image)
                image = Invert(image, 128)
                aug1 = Invert(aug1, 128)
                aug2 = Invert(aug2, 128)
                image = to_tensor(image)
                aug1 = to_tensor(aug1)
                aug2 = to_tensor(aug2)
                return (image, aug1, aug2), grapheme_root, consonant_diacritic, vowel_diacritic
        elif self.mode == "valid":
            image = to_tensor(image)
            return image, lb
        else:
            image = to_tensor(image)
            return image, self.images[idx]