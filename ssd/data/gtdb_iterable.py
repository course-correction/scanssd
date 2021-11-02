"""
Author: Abhisek Dey
Scalable Data reader for the GTDB dataset
Uses sliding windows to generate sub-images
"""

# from .config import HOME
import logging
from glob import glob
import os.path as osp

import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import cv2
import torch
from torch.utils import data
import imagesize
import math
import numpy as np
import argparse
import time
import torch.nn.functional as F
from torch.tensor import Tensor
import torchvision.transforms as transforms
from ScanSSD.ssd.utils.augmentations import GenerateWindows, PadTensor, DPRLToTensor


class GTDBDetection(data.IterableDataset):
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
            mean=(246, 246, 246),
            split="train",
            transform=None,
    ):

        self.root = args.dataset_root
        self.data_file = data_file
        self.split = split
        self.transform = transform

        self.size = args.model_type
        self.window = args.window
        self.stride = int(args.stride * args.window)

        if self.split == 'train':
            ann_files = []
            for line in open(self.data_file):
                ann_files.append(self.root + "/annotations/"
                                 + line.strip() + ".pmath")
            # Filter out missing/incorrect annotations
            self.ann_files = []
            for ann in ann_files:
                if osp.exists(ann):
                    self.ann_files.append(ann)

            self.ids = ['/'.join(ann_file.replace(".pmath", "").split('/')[-2:])
                        for ann_file in self.ann_files]
            self.image_files = [osp.join(self.root, 'images', idx + '.png') for idx in
                                self.ids]

        elif self.split == 'test':
            image_files = []
            if args.op_mode == 'dev':
                for line in open(self.data_file):
                    image_files.append(self.root + "/images/" + line.strip() + ".png")
            elif args.op_mode == 'pipeline':
                for line in open(osp.join(self.root, self.data_file)):
                    image_files.extend(glob(self.root + "/images/" + line.strip() + "/*.png"))
            else:
                raise Exception(f'Invalid output mode selected {args.op_mode}. '
                                f'Use \'dev\' or \'pipeline\'')
            self.image_files = image_files
            self.ids = ['/'.join(image_file.replace(".png", "").split('/')[-2:])
                        for image_file in self.image_files]
        else:
            raise Exception(f'Invalid mode selected: {self.split}')

        # Get the total windows
        windows = []
        tot = 0
        for path in self.image_files:
            width, height = imagesize.get(path)
            new_w = (math.ceil(width / self.window)) * self.window
            new_h = (math.ceil(height / self.window)) * self.window
            num_win_w = ((new_w - self.window) / self.stride) + 1
            num_win_h = ((new_h - self.window) / self.stride) + 1
            tot_page_win = num_win_w * num_win_h
            tot += tot_page_win
            windows.append(tot)

        self.tot = int(tot)
        self.windows = np.asarray(windows, dtype=int)
        self.start = 0
        self.end = len(self.image_files)
        
        # RZ: Removing for now.
        #logging.debug(f'Total Windows to be generated: {self.tot}')

        # Mean tensor for normalizing the images
        self.mean = torch.ones((3, self.size, self.size), device='cpu') * mean[0]

        self.transform = transforms.Compose([
            DPRLToTensor(),
            PadTensor(window=self.window),
            GenerateWindows(window=self.window, stride=self.stride)
        ]
        )

    def get_streams_test(self, start, end):
        return self.get_win_test(start, end)

    def get_streams_train(self, start, end):
        return self.get_win_train(start, end)

    def __iter__(self):
        if self.split == 'test':
            return self.get_streams_test(self.start, self.end)
        elif self.split == 'train':
            return self.get_streams_train(self.start, self.end)

    def get_win_test(self, start, end):

        for i in range(start, end):

            page_img = self.image_files[i]

            img_id = self.ids[i]
            img = cv2.imread(page_img)
            img_tensor = DPRLToTensor()(img).type('torch.FloatTensor')

            pad_h = ((math.ceil(img_tensor.shape[1] / self.window))
                     * self.window) - img_tensor.shape[1]
            pad_w = ((math.ceil(img_tensor.shape[2] / self.window))
                     * self.window) - img_tensor.shape[2]

            img_tensor = PadTensor(self.window)(img_tensor)
            img_windows = GenerateWindows(window=self.window, stride=self.stride)(img_tensor)


            # Yield the windows for the page
            for h in range(img_windows.shape[0]):
                for w in range(img_windows.shape[1]):
                    # Resize the window to model input shape
                    tgt_win = img_windows[h, w].unsqueeze(0)
                    tgt_win = F.interpolate(tgt_win, size=self.size,
                                            mode='area').squeeze()
                    tgt_win -= self.mean
                    # Yield testing data
                    yield h, w, pad_h, pad_w, tgt_win, img_id

    def get_win_train(self, start, end):

        for i in range(start, end):
            page_img = self.image_files[i]
            img = cv2.imread(page_img)
            img_id = self.ids[i]
            img_tensor = DPRLToTensor()(img).type('torch.FloatTensor')

            # Pad to perfectly fit windows
            # Get the pad for h and w
            pad_h = ((math.ceil(img_tensor.shape[1] / self.window))
                     * self.window) - img_tensor.shape[1]
            pad_w = ((math.ceil(img_tensor.shape[2] / self.window))
                     * self.window) - img_tensor.shape[2]

            # Get GT's
            all_gts = np.genfromtxt(self.ann_files[i], delimiter=',')
            all_gts_ten = torch.from_numpy(all_gts.copy()).type('torch.FloatTensor').reshape((-1, 4))
            # Add padding to all GT's
            try:
                all_gts_ten[:, [0, 2]] += int(pad_w / 2)
                all_gts_ten[:, [1, 3]] += int(pad_h / 2)
            except IndexError:
                logging.debug(f'Error in GT ann {self.ann_files[i]}')
                exit(0)

            img_tensor = PadTensor(self.window)(img_tensor)

            img_windows = GenerateWindows(window=self.window, stride=self.stride)(img_tensor)

            # Yield the windows for the page
            for h in range(img_windows.shape[0]):
                for w in range(img_windows.shape[1]):
                    tgt_win = img_windows[h, w].unsqueeze(0)
                    tgt_win = F.interpolate(tgt_win, size=self.size,
                                            mode='area').squeeze()

                    # Yield training data if in train mode
                    win_gts = self.get_gts(all_gts_ten, h, w)
                    # Skip windows if there are no GT's found for the window
                    if win_gts.shape[0] == 0:
                        continue
                    # TODO Have to change this for multi-classes
                    labels = torch.zeros(win_gts.shape[0], device='cpu')
                    # Normalize with respect to image mean
                    tgt_win -= self.mean
                    # tgt_win = tgt_win/255.
                    # print(tgt_win)
                    # print(tgt_win.type())
                    # exit(0)

                    labels = labels.reshape((-1, 1))
                    win_gts = torch.cat((win_gts, labels), dim=1)
                    # Yield training data
                    yield tgt_win, win_gts

    def get_gts(self, all_gts, h, w):
        base_offset = self.stride
        x_min = w * base_offset
        x_max = x_min + self.window
        y_min = h * base_offset
        y_max = y_min + self.window
        win_boxes = all_gts[(((all_gts[:, 2] >= x_min) & (all_gts[:, 0] <= x_max)) &
                             ((all_gts[:, 3] >= y_min) & (all_gts[:, 1] <= y_max)))]

        # Clip the boxes to x_min, y_min, x_max, y_max
        win_boxes[:, [0, 2]] = win_boxes[:, [0, 2]].clamp(min=x_min, max=x_max)
        win_boxes[:, [1, 3]] = win_boxes[:, [1, 3]].clamp(min=y_min, max=y_max)

        # Normalize the intersecting boxes between 0 and 1
        win_boxes[:, [0, 2]] = (win_boxes[:, [0, 2]] - x_min) / self.window
        win_boxes[:, [1, 3]] = (win_boxes[:, [1, 3]] - y_min) / self.window

        return win_boxes


def worker_init_fn(worker_id):
    worker_info = data.get_worker_info()
    dataset = worker_info.dataset
    overall_start = dataset.start
    overall_end = dataset.end
    per_worker = int(math.ceil((overall_end - overall_start) /
                               float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(overall_end, dataset.start + per_worker)


# Helper to verify functionality of new data loader
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test args for new dataset')
    parser.add_argument('--dataset_root',
                        default='/home/abhisek/Desktop/MathSeer-extraction-pipeline/modules/ScanSSD/gtdb_data',
                        type=str)
    parser.add_argument('--data_file', default='/home/abhisek/Desktop/MathSeer-extraction-pipeline'
                                               '/modules/ScanSSD/test_one_train', type=str)
    parser.add_argument('--stride', default=0.05, type=float)
    parser.add_argument('--window', default=1200, type=int)
    parser.add_argument('--model_type', default=512, type=int)
    args = parser.parse_args()

    dataset = GTDBDetection(args, args.data_file, split='train')
    loader = data.DataLoader(dataset, batch_size=1, num_workers=2, worker_init_fn=worker_init_fn)

    tot = dataset.tot
    tot_batches = math.ceil(tot / 3)
    tot_val_wins = 0
    tot_inval_wins = 0
    for b_id, (tgt_wins, win_gts) in enumerate(loader):
        # print(f'Batch id: {b_id}')
        # print(h_id)
        # print(w_id)
        # print(img_id)
        # print(img.shape)
        # print(b_id)
        print(tgt_wins.shape)
        print(win_gts.shape)

        # exit(0)
    # print(f'Total windows: {tot_val_wins+tot_inval_wins}')
    # print(f'% Invalid wins: {tot_inval_wins/(tot_val_wins+tot_inval_wins)}')
