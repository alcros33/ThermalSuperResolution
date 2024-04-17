import random
from PIL import Image
from pathlib import Path
import pandas as pd
import torch
import torchvision.transforms as tfms, torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import lightning as L
from functools import partial

from utils import random_split

def expand2square(img:Image.Image):
    width, height = img.size
    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), 0)
        result.paste(img, (0, (width - height) // 2))
        bbox = (0, (width - height) // 2, width, (width - height) // 2 + height)
        return result
    else:
        result = Image.new(img.mode, (height, height), 0)
        result.paste(img, ((height - width) // 2, 0))
        bbox = ((height - width) // 2, 0, (height - width) // 2 + width, height)
        return result

def pair_random_crop(img1, img2, size):
    i, j, h, w = tfms.RandomCrop.get_params(img1, size)
    return TF.crop(img1, i, j, h, w), TF.crop(img2, i, j, h, w)

def pair_random_horizontal_flip(img1, img2, p=0.5):
    if torch.rand(1) < p:
        return TF.hflip(img1), TF.hflip(img2)
    return img1, img2

def multi_img_random_horizontal_flip(*imgs, p=0.5):
    if torch.rand(1) < p:
        return tuple(TF.hflip(img) for img in imgs)
    return imgs

def pair_random_rotation(img1, img2, max_degrees, interpolation=tfms.InterpolationMode.NEAREST,
                              fill=0, expand=False, center=None):
    channels, _, _ = TF.get_dimensions(img1)
    if isinstance(img1, torch.Tensor):
        fill = [float(fill)] * channels
    angle = tfms.RandomRotation.get_params([-max_degrees,max_degrees])
    return TF.rotate(img1, angle, interpolation, expand, center, fill), TF.rotate(img2, angle, interpolation, expand, center, fill)

def pair_compose_augmentation(fns):
    def ret_val(img1, img2):
        for fn in fns:
            img1, img2 = fn(img1, img2)
        return img1, img2
    return ret_val

class SimpleImageDataset(Dataset):
    def __init__(self, img_files, transforms, mode="RGB"):
        self.img_files = img_files
        self.transforms = transforms
        self.mode = mode
    
    def __getitem__(self, index):
        img = Image.open(self.img_files[index]).convert(self.mode)
        return self.transforms(img)

    def __len__(self):
        return len(self.img_files)

class MultiImageDataset(Dataset):
    def __init__(self, imgs_files_list, transforms_list,
                 augmentations, mode="RGB"):
        assert len(imgs_files_list) == len(transforms_list)
        last = imgs_files_list[0]
        for im_list in imgs_files_list:
            assert len(im_list) == len(last)
            last = im_list
        self.dim = len(imgs_files_list)
        self.transforms = transforms_list
        self.img_files = imgs_files_list
        self.augmentations = augmentations
        self.mode = mode
    
    def __getitem__(self, index):
        img_list = tuple(Image.open(self.img_files[n][index]).convert(self.mode) for n in range(self.dim))
        if self.augmentations is not None:
            img_list = self.augmentations(*img_list)
        return tuple(tfm(img) for img, tfm in zip(img_list, self.transforms))

    def __len__(self):
        return len(self.img_files[0])

class SimpleImageDataModule(L.LightningDataModule):
    def __init__(self, data_dir:Path, batch_size: int = 8,
                 img_size=256, do_crop=False,
                 train_img_list=None, valid_img_list=None, test_img_list=None,
                 valid_pct=0.05, test_pct=0.05,
                 mean=0.5, std=0.5, img_mode="RGB"):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        
        if train_img_list is None:
            self.train_img_files = [f for f in data_dir.iterdir()]
        else:
            with open(train_img_list,'r') as f:
                self.train_img_files = [data_dir/fname[:-1] for fname in f]

        if test_img_list is None:
            self.train_img_files, self.test_img_files = random_split(self.train_img_files, test_pct)
        else:
            with open(test_img_list,'r') as f:
                self.test_img_files = [data_dir/fname[:-1] for fname in f]
        
        if valid_img_list is None:
            self.train_img_files, self.valid_img_files = random_split(self.train_img_files, valid_pct)
        else:
            with open(valid_img_list,'r') as f:
                self.valid_img_files = [data_dir/fname[:-1] for fname in f]

        self.mean, self.std = torch.tensor(mean), torch.tensor(std)
        self.transforms = tfms.Compose([
            tfms.Resize(img_size, interpolation=tfms.InterpolationMode.BICUBIC, antialias=True),
            tfms.RandomCrop(img_size) if do_crop else tfms.Lambda(expand2square),
            tfms.RandomHorizontalFlip(),
            tfms.RandomRotation(20),
            tfms.RandomAdjustSharpness(1.5, 0.3),
            tfms.ToTensor(),
            tfms.Normalize(self.mean, self.std),
            ])
        self.test_transforms = tfms.Compose([
            tfms.Resize(img_size, interpolation=tfms.InterpolationMode.BICUBIC, antialias=True),
            tfms.RandomCrop(img_size) if do_crop else tfms.Lambda(expand2square),
            tfms.ToTensor(),
            tfms.Normalize(self.mean, self.std),
            ])
        self.dataset_train = SimpleImageDataset(self.train_img_files, self.transforms, mode=img_mode)
        self.dataset_valid = SimpleImageDataset(self.valid_img_files, self.test_transforms, mode=img_mode)
        self.dataset_test = SimpleImageDataset(self.test_img_files, self.test_transforms, mode=img_mode)
    
    def setup(self, stage: str):
        pass
    
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=24)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size, num_workers=24)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=24)
    
    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=24)

class MultiImageDataModule(L.LightningDataModule):
    def __init__(self, data_dir:Path, batch_size: int = 8,
                 img_size=(256,256), num_workers=24,
                 splits:list[str]=("train", "val", "test"),
                 classes:list[str]=None,
                 formats=["jpg", "jpeg", "png", "bmp"],
                 mean=0.5, std=0.5):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        formats = formats + [f.upper() for f in formats]
        if classes is None:
            classes = ["."]
        
        train_img_files = list()
        for ext in formats:
            train_img_files.extend(list((data_dir/splits[0]/classes[0]).glob(f"**/*.{ext}")))
        train_img_files = [train_img_files] + [[data_dir/splits[0]/c/f.name
                                                for f in train_img_files] for c in classes[1:]]

        val_img_files = list()
        for ext in formats:
            val_img_files.extend(list((data_dir/splits[1]/classes[0]).glob(f"**/*.{ext}")))
        val_img_files = [val_img_files] + [[data_dir/splits[1]/c/f.name
                                            for f in val_img_files] for c in classes[1:]]
        
        if len(splits) >= 3:
            test_img_files = list()
            for ext in formats:
                test_img_files.extend(list((data_dir/splits[2]/classes[0]).glob(f"**/*.{ext}")))
            test_img_files = [test_img_files] + [[data_dir/splits[2]/c/f.name
                                                for f in test_img_files] for c in classes[1:-1]]
        else:
            test_img_files = val_img_files[:]

        self.mean, self.std = torch.tensor(mean), torch.tensor(std)
        transforms = tfms.Compose([
            tfms.ToTensor(),
            tfms.Normalize(self.mean, self.std)
            ])
        
        augments = multi_img_random_horizontal_flip
        
        self.dataset_train = MultiImageDataset(train_img_files, [transforms]*len(classes), augments)
        self.dataset_valid = MultiImageDataset(val_img_files, [transforms]*len(classes), None)
        self.dataset_test = MultiImageDataset(test_img_files, [transforms]*(len(classes)-1), None)
    
    def setup(self, stage: str):
        pass
    
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=24)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size, num_workers=24)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=24)
    
    def predict_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=24)
