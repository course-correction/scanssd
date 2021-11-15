import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
import sys
import argparse
import math


def draw_rects(img, data, color: tuple, thickness: int):
    for box in data:
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return img


def load_data(data_file, page_num):
    loaded_data = np.genfromtxt(data_file, delimiter=',')
    # if there is only one entry convert it to correct form required
    if len(loaded_data.shape) == 1:
        loaded_data = loaded_data.reshape(1, -1)
    loaded_data = loaded_data[loaded_data[:, 0] == page_num][:, 1:]
    return loaded_data.astype('int32')


def viz_boxes(pred_csv=None, gt_csv=None, page_num=1, img_path=''):
    if pred_csv:
        file_name = pred_csv.split('/')[-1].replace('.csv', '') + '_' + str(page_num)
    else:
        file_name = gt_csv.split('/')[-1].replace('.csv', '') + '_' + str(page_num)
    img = cv2.imread(img_path, 1)

    if pred_csv:
        pred_data = load_data(pred_csv, page_num)
        img = draw_rects(img, pred_data, (0, 255, 0), 10)

    if gt_csv:
        gt_data = load_data(gt_csv, page_num)
        img = draw_rects(img, gt_data, (255, 0, 0), 3)

    plt.imshow(img)
    # plt.show()
    plt.savefig(file_name + '.png', dpi=600)


def make_stride_windows(stride, img_path, gt_csv=None, page_num=None):
    image_path = img_path
    img = cv2.imread(image_path, 1)
    if gt_csv is not None:
        gt_file = gt_csv
        gt_data = np.genfromtxt(gt_file, delimiter=',')
        # if there is only one entry convert it to correct form required
        if len(gt_data.shape) == 1:
            gt_data = gt_data.reshape(1, -1)
        gt_data = gt_data[gt_data[:, 0] == page_num][:, 1:]

        for box in gt_data:
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)

    new_h = math.ceil(img.shape[0] / 1200) * 1200
    new_w = math.ceil(img.shape[1] / 1200) * 1200
    img = cv2.resize(img, (new_w, new_h))
    offset = stride * 1200

    window_ys = np.arange(0, img.shape[0], offset, dtype=np.int)
    window_xs = np.arange(0, img.shape[1], offset, dtype=np.int)

    for idx, y in enumerate(window_ys):
        for idx_t, window in enumerate(window_xs):
            img = cv2.rectangle(img, (window, y), ((window + 1200), (y + 1200)),
                                (0, 0, 255), 50)

    plt.imshow(img)
    plt.axis('off')
    plt.savefig('Test_Figures/stride' + str(stride) + '.png', bbox_inches='tight')


def adjust_pred_page_nums(dir):
    pdfs = glob.glob(dir + '*.csv')
    for pdf in pdfs:
        pred_data = np.genfromtxt(pdf, delimiter=',')
        # if there is only one entry convert it to correct form required
        if len(pred_data.shape) == 1:
            pred_data = pred_data.reshape(1, -1)

        page_nums = pred_data[:, 0].reshape((-1, 1))
        page_nums -= 1
        preds = pred_data[:, 1:]

        new_lines = np.concatenate((page_nums, preds), axis=1)
        pred_file = open(pdf, 'w')
        np.savetxt(pred_file, new_lines, fmt='%.2f', delimiter=',')


def viz_from_anno(img_path, anno_path):
    anno_data = np.genfromtxt(anno_path, delimiter=',').reshape((-1,4)).astype('int32')
    img = cv2.imread(img_path, 1)
    img = draw_rects(img, anno_data, (255,0,0), 3)
    plt.imshow(img)
    plt.show()


def make_eval_gt(data_file, anno_dir, eval_dir):
    '''
    Args:
        pdf_dir: Takes the path of the directory containing the .pmath files
        eval_dir: The output directory for writing the csv file for all the .pmath files
    '''
    file = open(data_file, 'r')
    insts = file.readlines()

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    for inst in sorted(insts):
        pdf_name = inst.strip().split('/')[0]
        page_num = int(inst.strip().split('/')[1])

        math_file = open(os.path.join(eval_dir, pdf_name + '.csv'), 'a')
        page_boxes = np.genfromtxt(os.path.join(anno_dir, inst.strip()+'.pmath'),
                                   delimiter=',').astype(np.int).reshape((-1,4))
        page_num_col = np.ones(len(page_boxes)).astype(
            np.int).reshape((-1, 1)) * (page_num-1)
        all_page_data = np.concatenate((page_num_col, page_boxes), axis=1)
        np.savetxt(math_file, all_page_data, fmt='%d', delimiter=',')
        math_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process visualization params.')
    parser.add_argument('--pagenum', type=int, help='page number')
    parser.add_argument('--imgpath', type=str, help='path to image to write over')
    parser.add_argument('--predcsv', default=None, type=str, help='predictions csv')
    parser.add_argument('--gtcsv', default=None, type=str, help='ground truth csv')
    parser.add_argument('--pmath', default=False, type=bool, help='Grnd Tth as pmath')
    parser.add_argument('--convert', default=False, type=bool, help='Make eval GT')
    parser.add_argument('--data_file', type=str, help='File List')
    parser.add_argument('--anno_dir', type=str, help='Directory of the .pmath files')
    parser.add_argument('--eval_dir', type=str, help='Output target dir for GT')

    args = parser.parse_args()
    if args.pmath or args.convert:
        if args.pmath:
            viz_from_anno(args.imgpath, args.gtcsv)
        else:
            make_eval_gt(args.data_file, args.anno_dir, args.eval_dir)
    else:
        viz_boxes(args.predcsv, args.gtcsv, args.pagenum, args.imgpath)
