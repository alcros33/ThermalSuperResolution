from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from time import perf_counter
import argparse
from diffusion_resshift import DiffuserSRResShift, F, make_grid
from unet_refiner import UnetRefiner
from utils import chunks

def open_img(fname):
    return TF.to_tensor(Image.open(fname))*2-1

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint')
parser.add_argument('--refiner', action='store_true')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--bs', default=4)
parser.add_argument('-d', '--base-dir', default='.')
parser.add_argument('-i', '--input-dir', nargs='+', default=[])
parser.add_argument('-o', '--output-dir', default="results")
        
if __name__ == "__main__":
    args = parser.parse_args()
    if args.refiner:
        model = UnetRefiner.load_from_checkpoint(args.checkpoint)
    else:
        model = DiffuserSRResShift.load_from_checkpoint(args.checkpoint)
    model.eval()
    if args.cuda:
        model.cuda()
    BASE_DIR = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    files = []
    n_inputs = len(args.input_dir)
    files = list((BASE_DIR/args.input_dir[0]).iterdir())
    files = [files] + [[BASE_DIR/folder/f.name for f in files]
                        for folder in args.input_dir[1:]]
    
    for fnames in chunks(list(zip(*files)), args.bs):
        batch = []
        for i in range(n_inputs):
            batch.append(torch.stack([open_img(fname[i]).cuda()
                                    if args.cuda else open_img(fname[i])
                                    for fname in fnames]))
        batch = tuple(batch)
        pred = model.predict_step(batch, None)*0.5+0.5
        for it, fname in enumerate(fnames):
            TF.to_pil_image(pred[it]).save(output_dir/fname[0].name)
    print("DONE")
    