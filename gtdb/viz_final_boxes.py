import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
import sys
import math


def viz_boxes():
    pred = 'eval/Chem_Run1/raw_output/US20050043351A1.csv'
    # pred_file = glob.glob(pred + '/*.csv')
    gt_file = 'eval/chem_gt_small/US20050043351A1.csv'

    pred_data = np.genfromtxt(pred, delimiter=',')
    # if there is only one entry convert it to correct form required
    if len(pred_data.shape) == 1:
           pred_data = pred_data.reshape(1, -1)

    gt_data = np.genfromtxt(gt_file, delimiter=',')
    # if there is only one entry convert it to correct form required
    if len(gt_data.shape) == 1:
           gt_data = gt_data.reshape(1, -1)
    gt_data = gt_data[gt_data[:,0] == 28][:,1:]
    #
    pred_data = pred_data[pred_data[:,0] == 28][:,1:]

    image_path = '/home/abhisek/Desktop/MathSeer-extraction-pipeline/src/ScanSSD' \
                 '/chem_data/images/US20050043351A1/29.png'
    img = cv2.imread(image_path, 1)

    pred_data = pred_data.astype('int32')
    gt_data = gt_data.astype('int32')

    for box in pred_data:
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 10)

    # ground truth is red
    for box in gt_data:
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 5)

    # img = cv2.resize(img, (1000,1200))

    plt.imshow(img)
    # plt.axis('off')
    plt.show()
    # plt.savefig('Test_Figures/Chem_Test1.png', dpi=300, bbox_inches='tight')


def make_stride_windows(stride):
    image_path = '/home/abhisek/Desktop/MathSeer-extraction-pipeline/src/ScanSSD/chem_data' \
                 '/images/US20050043351A1/69.png'
    gt_file = 'eval/chem_gt_small/US20050043351A1.csv'
    gt_data = np.genfromtxt(gt_file, delimiter=',')
    # if there is only one entry convert it to correct form required
    if len(gt_data.shape) == 1:
           gt_data = gt_data.reshape(1, -1)
    gt_data = gt_data[gt_data[:,0] == 1][:,1:]
    img = cv2.imread(image_path, 1)
    # for box in gt_data:
    #     img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)

    new_h = math.ceil(img.shape[0]/1200) * 1200
    new_w = math.ceil(img.shape[1]/1200) * 1200
    img = cv2.resize(img, (new_w, new_h))
    offset = stride * 1200

    window_ys = np.arange(0,img.shape[0], offset, dtype=np.int)
    window_xs = np.arange(0,img.shape[1], offset, dtype=np.int)

    for idx, y in enumerate(window_ys):
        for idx_t, window in enumerate(window_xs):
            img = cv2.rectangle(img, (window, y), ((window+1200), (y+1200)),
                                (0,0,255), 50)
    # Add the center square rectangle
    win_center_x = window_xs[1]
    win_center_y = window_ys[1]
    img = cv2.rectangle(img, (win_center_x, win_center_y),
                        (win_center_x+1200, win_center_y+1200), (255,0,0), 50)

    plt.imshow(img)
    # plt.show()
    plt.axis('off')
    plt.savefig('Test_Figures/Chem_1.0_stride.png', bbox_inches='tight', dpi=600)


def adjust_pred_page_nums(dir):
    pdfs = glob.glob(dir + '*.csv')
    for pdf in pdfs:
        pred_data = np.genfromtxt(pdf, delimiter=',')
        # if there is only one entry convert it to correct form required
        if len(pred_data.shape) == 1:
            pred_data = pred_data.reshape(1, -1)

        page_nums = pred_data[:,0].reshape((-1,1))
        page_nums -= 1
        preds = pred_data[:,1:]

        new_lines = np.concatenate((page_nums, preds), axis=1)
        pred_file = open(pdf, 'w')
        np.savetxt(pred_file, new_lines, fmt='%.2f', delimiter=',')


def make_eval_gt(pdf_dir, eval_dir):
    '''
    Args:
        pdf_dir: Takes the path of the directory containing the .pmath files
        eval_dir: The output directory for writing the csv file for all the .pmath files
    '''
    pdf_name = pdf_dir.split('/')[-1]
    math_file = open(os.path.join(eval_dir,pdf_name+'.csv'), 'a')
    for gt_file in glob.glob(os.path.join(pdf_dir, '*.pmath')):
        page_num = int(gt_file.split('/')[-1].replace('.pmath', '')) - 1
        page_boxes = np.genfromtxt(gt_file, delimiter=',').astype(np.int).reshape((-1,4))
        page_num_col = np.ones(len(page_boxes)).astype(np.int).reshape((-1,1)) * page_num
        all_page_data = np.concatenate((page_num_col, page_boxes), axis=1)
        np.savetxt(math_file, all_page_data, fmt='%d', delimiter=',')
    math_file.close()


def chem_gts_convert():
    test_pdf_files = []
    for row in open('../chem_data_test_small', 'r'):
        test_pdf_files.append(row.split('/')[0])
    test_pdf_files = np.asarray(test_pdf_files)
    test_pdf_files = np.unique(test_pdf_files)
    base_path = '../chem_data/annotations'
    eval_dir = 'eval/chem_gt_small'
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    for pdf in test_pdf_files:
        pdf_dir = os.path.join(base_path, pdf)
        make_eval_gt(pdf_dir, eval_dir)


if __name__=='__main__':
    # viz_boxes()
    make_stride_windows(1.0)
    # adjust_pred_page_nums('eval/Merged_Test2/Merged_Test2/raw_output/')
    # make_eval_gt('../gtdb_data/annotations/Borcherds86', 'eval/math_gt')
    # chem_gts_convert()
