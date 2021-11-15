"""
Author: Abhisek Dey
Scalable Data reader for the GTDB dataset
Uses sliding windows to generate sub-images
"""

# from .config import HOME
from glob import glob
import os.path as osp
import sys

import cv2
import torch
import torch.utils.data as data
import imagesize
import math
import numpy as np
import argparse
import time
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.utils import box_utils
# from gtdb import feature_extractor
# import copy
# import utils.visualize as visualize



# GTDB_CLASSES = "math"  # always index 0 is background
#
# GTDB_ROOT = osp.join(HOME, "data_loaders/GTDB/")


class GTDBDetection(data.Dataset):
    """GTDB Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to GTDB folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name: `GTDB`
    """

    def __init__(
        self,
        args,
        data_file,
        mean=(246,246,246),
        split="train",
    ):

        self.root = args.dataset_root
        self.data_file = data_file
        self.split = split
        self.mean = mean

        self.size = args.model_type
        self.window = args.window
        self.stride = int(args.stride * args.window)


        image_files = []
        for line in open(osp.join(self.root, self.data_file)):
            image_files.extend(glob(self.root + "/images/" + line.strip() + ".png"))

        self.image_files = image_files
        self.ids = ['/'.join(image_file.replace(".png", "").split('/')[-2:])
                    for image_file in image_files]
        # GT Anns
        if self.split == 'train':
            ann_files = []
            for line in open(osp.join(self.root, self.data_file)):
                ann_files.extend(glob(self.root + "/annotations/"
                                      + line.strip() + ".pmath"))
            self.ann_files = ann_files

        # Get the total windows
        start = time.time()
        windows = []
        tot = 0
        for path in self.image_files:
            width, height = imagesize.get(path)
            new_w = (math.ceil(width / self.window)) * self.window
            new_h = (math.ceil(height / self.window)) * self.window
            num_win_w = ((new_w - self.window)/self.stride) + 1
            num_win_h = ((new_h - self.window)/self.stride) + 1
            tot_page_win = num_win_w * num_win_h
            tot += tot_page_win
            windows.append(tot)

        self.tot = int(tot)
        self.windows = np.asarray(windows, dtype=int)

    def __getitem__(self, index):
        if self.split == 'test':
            return self.get_win(index)
        else:
            print('Test not implemented')
            exit(0)
            # win, img_id, gt, offset_h, offset_w = self.get_win_gt(index)
            # return win, img_id, gt, offset_h, offset_w

    def __len__(self):
        return self.tot

    def get_win(self, index):

        for idx, win_count in enumerate(self.windows):
            if index <= win_count:
                page_id = idx
                break
        if page_id == 0:
            win_id = index
        else:
            win_id = index - self.windows[page_id-1]

        img = cv2.imread(self.image_files[page_id])
        img_id = self.ids[page_id]
        img_tensor = torch.from_numpy(img).permute(2,0,1)

        # Pad to perfectly fit windows
        # Get the pad for h and w
        pad_h = ((math.ceil(img_tensor.shape[1] / self.window))
                 * self.window) - img_tensor.shape[1]
        new_h = img_tensor.shape[1] + pad_h
        pad_w = ((math.ceil(img_tensor.shape[2] / self.window))
                 * self.window) - img_tensor.shape[2]
        new_w = img_tensor.shape[2] + pad_w

        # Make a new img_tensor with padding
        target_size_tensor = torch.ones((3, new_h, new_w), device='cpu') * 255
        target_size_tensor[:, int(pad_h / 2):int(pad_h / 2) + img_tensor.shape[1],
            int(pad_w / 2):int(pad_w / 2) + img_tensor.shape[2]] = img_tensor
        img_tensor = target_size_tensor

        img_windows = img_tensor.unfold(1, self.window, self.stride) \
            .unfold(2, self.window, self.stride).permute(1, 2, 0, 3, 4)
        h_id = max(0, math.ceil(win_id/img_windows.shape[1]) - 1)
        w_id = ((win_id % img_windows.shape[1]) +
                (img_windows.shape[1]-1)) % img_windows.shape[1]

        # Resize the window to model input shape
        tgt_win = img_windows[h_id, w_id].unsqueeze(0)
        tgt_win = F.interpolate(tgt_win, size=self.size,
                                mode='area').squeeze()

        return h_id, w_id, pad_h, pad_w, tgt_win, img_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test args for new dataset')
    parser.add_argument('--dataset_root',
                        default='/home/abhisek/Desktop/MathSeer-extraction-pipeline/modules/ScanSSD/gtdb_data',
                        type=str)
    parser.add_argument('--data_file', default='/home/abhisek/Desktop/MathSeer-extraction-pipeline'
                                               '/modules/ScanSSD/testing_data', type=str)
    parser.add_argument('--stride', default=1.0, type=float)
    parser.add_argument('--window', default=1200, type=int)
    parser.add_argument('--model_type', default=512, type=int)
    args = parser.parse_args()

    dataset = GTDBDetection(args, args.data_file, split='test')
    loader = data.DataLoader(dataset, batch_size=3, num_workers=2)

    for b_id, (h_id, w_id, img, img_id) in enumerate(loader):
        print(f'Batch id: {b_id}')
        print(h_id)
        print(w_id)
        print(img_id)
        # print(window.shape)
        if b_id == 20:
            break
