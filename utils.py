import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def dataset_stats(dataset, device="cuda:0", batch_size=32, shuffle=True, num_workers=24):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    mean = 0.
    std = 0.
    for _,images in dataloader:
        images = images.to(device)
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(dataset)
    std /= len(dataset)
    return mean, std

def linear_map(x:torch.tensor, min:float, max:float, new_min=0.0, new_max=1.0):
    return (x-min)/(max-min)*(new_max-new_min) + new_min

def num_params(module):
    n = 0
    for m in module.parameters():
        if m.requires_grad:
            n += m.numel()
    return n

def denorm(img, mean, std):
    return img*std[None][...,None,None] + mean[None][...,None,None]

def random_split(data:list, valid_pct=0.1):
    N = len(data)
    train_size = int(valid_pct*N)
    idx = torch.randperm(N)
    train_idx, valid_idx = idx[:train_size], idx[train_size:]
    return [data[i] for i in train_idx], [data[i] for i in valid_idx]

def chunks(l, n: int, reflect: bool = False):
    "Yield successive `n`-sized chunks from `l`."
    for i in range(0, len(l), n):
        if i+n > len(l) and reflect:
            yield l[i:i + n] + l[:(i+n)-len(l)]
        else:
            yield l[i:i + n]