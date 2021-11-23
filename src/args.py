import argparse
from typing import Tuple, List

import torch
import os
import logging

from src.data_loaders import GTDB_ROOT


def parse_test_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ScanSSD: Scanning Single Shot MultiBox Detector")
    parser.add_argument(
        "--trained_model",
        default="AMATH512_e1GTDB.pth",
        type=str,
        help="Trained state_dict file path to open",
    )
    parser.add_argument("--models_folder", default=None, type=str,
                        help='Optional folder path to test multiple checkpoints')
    parser.add_argument(
        "--save_folder", default="eval/", type=str, help="Dir to save results"
    )
    parser.add_argument(
        "--visual_threshold", default=0.6, type=float, help="Final confidence threshold"
    )
    parser.add_argument(
        "--cuda", default=False, type=bool, help="Use cuda to train model"
    )
    parser.add_argument(
        "--dataset_root", default="../", help="Location of VOC root directory"
    )
    parser.add_argument("--test_data", default="testing_data", help="testing data_loaders file")
    parser.add_argument("--verbose", default=False, type=bool, help="plot output")
    parser.add_argument(
        "--suffix",
        default="_10",
        type=str,
        help="suffix of directory of images for testing",
    )
    parser.add_argument(
        "--exp_name",
        default="SSD",
        help="Name of the experiment. Output will be saved at [save_folder]/[exp_name]/raw_output/",
    )
    parser.add_argument(
        "--model_type",
        default=300,
        type=int,
        help="Type of src model, ssd300 or ssd512",
    )
    parser.add_argument(
        "--use_char_info",
        default=False,
        type=bool,
        help="Whether or not to use char info",
    )
    parser.add_argument(
        "--limit", default=-1, type=int, help="limit on number of test examples"
    )
    parser.add_argument(
        "--cfg",
        default="gtdb",
        type=str,
        help="Type of network: either gtdb or math_gtdb_512",
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers",
        default=16,
        type=int,
        help="Number of workers used in data_loaders loading",
    )
    parser.add_argument(
        "--kernel",
        default=[1,5],
        type=int,
        nargs="+",
        help="Kernel size for feature layers: 3 3 or 1 5",
    )
    parser.add_argument(
        "--padding",
        default=[0,2],
        type=int,
        nargs="+",
        help="Padding for feature layers: 1 1 or 0 2",
    )
    parser.add_argument(
        "--neg_mining",
        default=True,
        type=bool,
        help="Whether or not to use hard negative mining with ratio 1:3",
    )
    parser.add_argument(
        "--log_dir", default="logs", type=str, help="dir to save the logs"
    )
    parser.add_argument(
        "--stride", default=0.1, type=float, help="Stride to use for sliding window"
    )
    parser.add_argument("--window", default=512, type=int, help="Sliding window size")
    parser.add_argument("--post_process", default=0, type=int,
                        help="Add cropping to predicted boxes")
    parser.add_argument("--op_mode", default='dev', type=str,
                        help='dev or pipeline (for extraction pipeline)')
    parser.add_argument("--conf", nargs='+', type=float, required=True,
                        help='Confidence for window lvl')
    parser.add_argument('--gpu', nargs='+', required=True, type=int,
                        help='GPU IDS to train on')
    parser.add_argument(
        "-f",
        default=None,
        type=str,
        help="Dummy arg so we can load in Jupyter Notebooks",
    )

    # if args.cuda and torch.cuda.is_available():
    #     torch.set_default_tensor_type("torch.cuda.FloatTensor")
    # else:
    #     torch.set_default_tensor_type("torch.FloatTensor")

    return parser


def parse_train_args():
    '''
    Read arguments and initialize directories
    :return: args
    '''
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')
    # train_set = parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset', default='GTDB', choices=['GTDB'],
                        type=str, help='choose GTDB')
    parser.add_argument('--dataset_root', default=GTDB_ROOT,
                        help='Dataset root directory path')
    parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                        help='Pretrained base model')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='Resume training at this iter')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in data_loaders loading')
    parser.add_argument('--cuda', default=False, type=bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='Alpha for the multibox loss')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--visdom', default=False, type=bool,
                        help='Use visdom for loss visualization')
    parser.add_argument('--exp_name', default='math_detector',  # changed to exp_name from --save_folder
                        help='It is the name of the experiment. Weights are saved in the directory with same name.')
    parser.add_argument('--layers_to_freeze', default=20, type=float,
                        help='Number of VGG16 layers to freeze')
    parser.add_argument('--model_type', default=300, type=int,
                        help='Type of src model, ssd300 or ssd512')
    parser.add_argument('--suffix', default="_10", type=str,
                        help='Stride % used while generating images or dpi from which images was generated or some other identifier')
    parser.add_argument('--training_data', default="training_data", type=str,
                        help='Training data_loaders to use. This is list of file names, one per line')
    parser.add_argument('--validation_data', default="", type=str,
                        help='Validation data_loaders to use. This is list of file names, one per line')
    parser.add_argument('--use_char_info', default=False, type=bool,
                        help='Whether to use char position info and labels')
    parser.add_argument('--cfg', default="ssd512", type=str,
                        help='Type of network: either gtdb or math_gtdb_512')
    parser.add_argument('--loss_fun', default="fl", type=str,
                        help='Type of loss: either fl (focal loss) or ce (cross entropy)')
    parser.add_argument('--kernel', type=int, nargs='+', default="3 3",
                        help='Kernel size for feature layers: 3 3 or 1 5')
    parser.add_argument('--padding', type=int, nargs='+', default="1 1",
                        help='Padding for feature layers: 1 1 or 0 2')
    parser.add_argument('--neg_mining', default=False, type=bool,
                        help='Whether or not to use hard negative mining with ratio 1:3')
    parser.add_argument('--log_dir', default=os.path.join("src", "logs"), type=str,
                        help='dir to save the logs')
    parser.add_argument('--stride', default=0.1, type=float,
                        help='Stride to use for sliding window')
    parser.add_argument('--window', default=300, type=int,
                        help='Sliding window size')
    parser.add_argument('--gpu', nargs='+', default='0', type=int, help='GPU IDS to train on')
    parser.add_argument('--pos_thresh', default=0.5, type=float,
                        help='All default boxes with iou>pos_thresh are considered as positive examples')

    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            logging.warning("WARNING: It looks like you have a CUDA device, but aren't " +
                            "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists("src/weights_" + args.exp_name):
        os.mkdir("src/weights_" + args.exp_name)

    return args


def get_gpus() -> Tuple[List[str], int, int]:
    free = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free').read().strip().split('\n')
    memory_available = [int(f.split()[2]) for f in free]
    gpus = []

    # AD: Get the min memory from all gpus as the threshold
    min_mem = min(memory_available)
    for i, memory in enumerate(memory_available):
        gpus.append(str(i))

    if min_mem < 6000:
        logging.warning('WARNING!! Your GPU might not have enough VRAM '
                        'to run this pipeline')
    per_gpu_batch_size = max(1, int((min_mem - 5500) / 500))

    num_gpus = len(gpus)
    return gpus, num_gpus, per_gpu_batch_size