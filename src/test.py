"""
This file contains functions to test and save the results
"""
import os
import glob
import os.path as osp
import argparse
import torch.backends.cudnn as cudnn
import torch.cuda.nccl
#import progressbar
from args import parse_test_args
from src.ssd import build_ssd
import logging
import time
import datetime
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from data_loaders.gtdb_iterable import worker_init_fn
from src.data_loaders import *
import shutil
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from src.utils.process_page import process_page


# uncomment this to hide the package deprecation warning 
# that may pop up depending on your version of torch
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def test_net_batch(args, net, dataset, devices, file_name):
    """
    Batch testing
    """

    # Creaet data_loaders loader and report workload.
    data_loader = DataLoader(
        dataset,
        args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    total = dataset.tot
    base_offset = int(args.window * args.stride)
    all_page_windows = {page_id+'_'+str(conf): [] for page_id in dataset.ids
                        for conf in args.conf}

    # Count non-blank lines to get documents in input file list.
    docs = 0
    infile = open( args.test_data, 'r' )
    Lines = infile.readlines()
    for line in Lines:
        if line.strip():
            docs += 1

    logging.debug("\n  Documents: " + str(docs) + "  Pages: " + str( len(dataset.ids) ) + \
            "  Windows: {}".format(total))
    logging.debug("  Performing window-level formula detection")

    windows_finished = 0
    page_times_list = []

    pbar = tqdm(total=dataset.tot, desc="  Processing")
    detect_start = time.time()
    for batch_idx, (h_ids, w_ids, pad_hs, pad_ws, windows, page_ids) in \
            enumerate(data_loader):

        start = time.time()

        if args.cuda:
            images = windows.to(devices)
            # targets = [ann.cuda() for ann in targets]
        else:
            images = Variable(windows)
            # targets = [Variable(ann, volatile=True) for ann in targets]

        # Detect and score candidate formula regions
        y = net(images)  # forward pass
        detections = y.data.cpu().numpy()

        # Convert all metadata to numpy (these are NOT in GPU, so no gpu-cpu)
        h_ids = h_ids.numpy()
        w_ids = w_ids.numpy()
        pad_hs = pad_hs.numpy()
        pad_ws = pad_ws.numpy()

        # Threshold detections
        for idx, (h_id, w_id, pad_h, pad_w, page_id) in \
                enumerate(zip(h_ids, w_ids, pad_hs, pad_ws, page_ids)):

            #print(idx, h_id, w_id, pad_h, pad_w, page_id)
            img_id = page_id
            y_l = h_id * base_offset
            x_l = w_id * base_offset

            pad_offset_x = int(pad_w/2)
            pad_offset_y = int(pad_h/2)

            for conf in args.conf:
                img_detections = detections[idx, 1, :, :]
                # Thredshold detections
                recognized_scores = img_detections[:,0]
                valid_mask = recognized_scores >= conf
                recognized_scores = recognized_scores[valid_mask]
                
                if len(recognized_scores):
                    recognized_boxes = img_detections[:,1:] * args.window
                    recognized_boxes = recognized_boxes[valid_mask]
                    # recognized_boxes_mask = (recognized_boxes[:,0]>0) & (recognized_boxes[:,1]>0) & (recognized_boxes[:,2]>0) & (recognized_boxes[:,3]>0) & (recognized_boxes[:,0]<=1) & (recognized_boxes[:,1]<=1) & (recognized_boxes[:,2]<=1) & (recognized_boxes[:,3]<=1)
                    # recognized_boxes = recognized_boxes[recognized_boxes_mask]
                    # recognized_scores = recognized_scores[recognized_boxes_mask]

                    # recognized_boxes = recognized_boxes * args.window
                    offset = np.array([x_l, y_l, x_l, y_l])
                    pad_offset = np.array([pad_offset_x, pad_offset_y,
                                           pad_offset_x, pad_offset_y])
                    recognized_boxes = recognized_boxes + offset - pad_offset

                    all_page_windows[img_id+'_'+str(conf)].\
                            append([recognized_boxes, recognized_scores])

        batch_time = time.time() - start
        page_times_list.append(batch_time)

        pbar.update( len(windows) )

    # Close the progress bar.
    pbar.close()

    print('  Avg window processing time: {:.2f}  seconds'
        .format((sum(page_times_list)/len(page_times_list))/args.batch_size))
    print('  Total processing time: {:.2f} seconds' 
            .format(time.time() - detect_start))

    # Apply XY and save final boxes
    process_page(args, all_page_windows, docs, file_name)


# RZ: renamed 'test_gtdb()' to 'detect()'
def detect(args):
    # Configure GPU utilization (passed in via args)
    # Assumes CUDA.
    gpus = [str(gpu_id) for gpu_id in args.gpu]
    gpus = ','.join(gpus)
    devices = torch.device('cuda:' + gpus)

    # Create network inputs from quick_start_data
    dataset = GTDBDetection(
        args,
        args.test_data,
        mean=(246, 246, 246),
        split="test",
    )

    # Initialize neural network
    num_classes = 2  # +1 background
    net = build_ssd(
        args, "test", exp_cfg[args.cfg], devices, args.model_type, num_classes
    )
    net = nn.DataParallel(net, device_ids=args.gpu)

    # logging.debug(net)
    # net.to(devices)
    # NOTE: if things aren't working, you may need to replace the previous
    # uncommented line with the previous two lines.


    # Collect weight sets (models) to use, from passed folder or
    # weight file list.
    if args.models_folder is None:
        checkpoints = [args.trained_model]
        file_names = [args.trained_model.split('/')[-1].replace('.pth','')]
    else:
        checkpoints = sorted(glob.glob(args.models_folder + '/*'))
        file_names = [ckpt.split('/')[-1].replace('.pth', '')
                            for ckpt in checkpoints]

    # Run the network for each set of network weights (model)
    for idx, checkpoint in enumerate(checkpoints):
        logging.debug(f'  Loading weights: '
                      f'{file_names[idx]}')

        # Configure the network, set network to 'eval'uation mode.
        net.module.load_state_dict(
            torch.load(
                checkpoint,
            )
        )
        net.eval()
        if args.cuda:
            net = net.to(devices)
            cudnn.benchmark = True

        # Run the network, threshold and merge detections,
        # Generate output.
        # RZ: passing filenames in a list.
        test_net_batch(
            args,
            net,
            dataset,
            devices,
            file_names[idx]
        )


if __name__ == "__main__":

    # Process arguments
    start = time.time()
    args = parse_test_args().parse_args()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    if os.path.exists(os.path.join(args.save_folder, args.exp_name, "raw_output")):
        shutil.rmtree(os.path.join(args.save_folder, args.exp_name, "raw_output"))

    try:
        # Output execution information.
        print("\n[ ScanSSD ]")
        print("  Image dir: " + args.dataset_root )
        print("  Output (CSV) dir: " + os.path.abspath(args.save_folder) )

        # Set up logging.
        filepath = os.path.join(
            args.log_dir, args.exp_name + "_" + str(round(time.time())) + ".log"
        )

        # Share log file and file list.
        print("  Log file: " + osp.abspath(filepath))
        print("  File list: " + args.test_data )

        logging.basicConfig(
            filename=filepath,
            filemode="w",
            format="%(process)d - %(asctime)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            level=logging.DEBUG,
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.getLogger('PIL').setLevel(logging.INFO)

        # Run detection.
        detect(args)

    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        print("Exception occurred: \n{}".format(e))

    # Report completion, execution time.
    end = time.time()
    logging.debug("\n  Detection time: {:.2f} seconds".\
            format( time.time() - start))
