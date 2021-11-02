import glob
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv

CHAR_CT_DICT = {p:0 for p in range(4)}
MATCHES_DICT = {p:0 for p in range(5)}

def draw_rects(img, data, color: tuple, thickness: int):
    for box in data:
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return img


def collate_symbols(data, page_data, page_num, math_rows):
    num_syms = len(math_rows)
    if 0 < num_syms < 4:
        sym_id = num_syms
    else:
        sym_id = 0
    for_boxes = []
    for i in range(math_rows[0], math_rows[-1] + 1):
        for_boxes.append(data[i][2:6])
    for_boxes = np.array(for_boxes, dtype=np.float).reshape((-1, 4))
    x_min, x_max = np.min(for_boxes[:, 0]), np.max(for_boxes[:, 2])
    y_min, y_max = np.min(for_boxes[:, 1]), np.max(for_boxes[:, 3])
    page_data[page_num][sym_id].append([x_min, y_min, x_max, y_max])
    return page_data

def extract_char_data(write=False, hist=False, compare=False):
    char_files_folder = 'gtdb_data/testing_data_char/'
    char_files = glob.glob(os.path.join(char_files_folder,'*.csv'))
    annotation_folder = 'ssd/eval/'

    for char_file in char_files:
        pdf_name = char_file.split('/')[-1].replace('.csv', '')
        data = []
        with open(char_file, newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                data.append(row)

        page_data = {}
        # Get num pages
        for row in data:
            if row[0] == 'Sheet':
                page_data[int(row[1])] = {p:[] for p in range(4)}

        page_num = 0
        math_rows = []
        for row_id, row in enumerate(data):
            if row[0] == 'Sheet':
                if len(math_rows):
                    page_data = collate_symbols(data, page_data, page_num, math_rows)
                    math_rows = []
                page_num = int(row[1])
            elif row[0] == 'Chardata':
                if row[6] == '1':
                    math_rows.append(row_id)
                elif row[6]=='0'  and len(math_rows) > 0 and row_id>4:
                    page_data = collate_symbols(data, page_data, page_num, math_rows)
                    math_rows = []
            elif row[0]!='Chardata' and len(math_rows):
                page_data = collate_symbols(data, page_data, page_num, math_rows)
                math_rows = []
            elif row_id == len(data)-1 and len(math_rows):
                math_rows.append(row_id)
                page_data = collate_symbols(data, page_data, page_num, math_rows)
                math_rows = []

        if write:
            write_char_data(page_data, annotation_folder, pdf_name)
        if hist:
            count_chars(page_data, CHAR_CT_DICT)
        if compare:
            compare_anns(page_data, pdf_name, MATCHES_DICT)
        # if pdf_name == 'Li75':
        #     viz_char_formula_anns(page_data, pdf_name)
    if hist:
        print(CHAR_CT_DICT)
        plot_hist(CHAR_CT_DICT)
    if compare:
        print(MATCHES_DICT)


def write_char_data(page_data, annotation_folder, pdf_name):
    for page in sorted(page_data.keys()):
        for sym_id in page_data[page].keys():
            symbols = sym_id
            if 0 < symbols < 20:
                csv_file = os.path.join(annotation_folder,'math_gt_'+str(sym_id)+'_symbols',
                                        pdf_name+'.csv')
            else:
                csv_file = os.path.join(annotation_folder,'math_gt_'+'large'+'_symbols',
                                        pdf_name+'.csv')
            if not os.path.exists(os.path.dirname(csv_file)):
                os.makedirs(os.path.dirname(csv_file))

            boxes = np.array(page_data[page][sym_id], dtype=np.int).reshape((-1,4))
            num_pages = len(boxes)
            write_data = np.concatenate((np.full((num_pages,1), page, dtype=np.int),
                                         boxes), axis=1)
            math_file = open(csv_file, 'a')
            np.savetxt(math_file, write_data, fmt='%d', delimiter=',')
            math_file.close()


def count_chars(doc_data, char_ct_dict):
    for page in doc_data.keys():
        for num_syms in doc_data[page].keys():
            char_ct_dict[num_syms] += len(doc_data[page][num_syms])


def viz_char_ann(page_data):
    # Test Emden76 Page 1 or 2
    img_file = 'gtdb_data/images/Emden76/1.png'
    img = cv2.imread(img_file, 1)
    page_box_dict = page_data[1]
    formula_type = [2]
    for num_sym in page_box_dict.keys():
        if len(page_box_dict[num_sym]) > 0:
            boxes = np.array(page_box_dict[num_sym], dtype=np.int).reshape((-1,4))
            img = draw_rects(img, boxes, (0, 255, 0), 3)
    plt.imshow(img)
    plt.show()


def compare_anns(doc_data, pdf_name, matches_dict, form_dict):
    pages = sorted(list(doc_data.keys()))
    for page in pages:
        page_boxes = []
        for num_sym in doc_data[page].keys():
            char_ann = np.array(doc_data[page][num_sym], dtype=np.int).reshape((-1, 4))
            if len(char_ann):
                for box in char_ann:
                    page_boxes.append(box)
        page_boxes = np.array(page_boxes).reshape((-1,4))
        orig_page_ann = np.array(form_dict[page], dtype=np.int).reshape((-1,4))
        page_boxes = page_boxes[np.lexsort((page_boxes[:,0],page_boxes[:,1],
                                            page_boxes[:,2],page_boxes[:,3]))]
        orig_page_ann = orig_page_ann[np.lexsort((orig_page_ann[:,0],orig_page_ann[:,1],
                                                  orig_page_ann[:,2],orig_page_ann[:,3]))]
        if len(page_boxes) == len(orig_page_ann):
            for idx in range(len(page_boxes)):
                diff = np.sum(np.abs(page_boxes[idx]-orig_page_ann[idx]))
                if diff == 0:
                    matches_dict[0] += 1
                elif diff <= 4:
                    matches_dict[1] += 1
                elif diff <= 10:
                    matches_dict[2] += 1
                elif diff <=50:
                    matches_dict[3] += 1
                else:
                    matches_dict[4] += 1

        else:
            print(f'Ann mismatch PDF {pdf_name} page {page}')

def plot_hist(char_ct_dict, num_bins):
    # Define label headers
    labels = []
    for i in range(num_bins):
        if i == 0:
            labels.append(str(num_bins)+'+ symbols')
        else:
            labels.append(str(i)+' symbols')

    # Create counts array
    counts = []
    for ct in sorted(char_ct_dict.keys()):
        counts.append(char_ct_dict[ct])
    sorted_labels = labels[1:]
    sorted_labels.append(labels[0])
    sorted_counts = counts[1:]
    sorted_counts.append(counts[0])
    plt.bar(sorted_labels, sorted_counts, label='Symbol Counts')
    plt.title('1 vs 2 vs ... vs'+str(num_bins)+'+ symbols in GTDB Test Set')
    plt.xlabel('Symbols in each formula')
    plt.ylabel('Counts')
    x = np.arange(len(sorted_labels))
    plt.xticks(x, sorted_labels)
    plt.show()

    labels = ['1/2/3... symbol', str(num_bins)+ '+ symbols']
    sm_counts = 0
    for ct in sorted_counts[:-1]:
        sm_counts += ct
    l_counts = sorted_counts[-1]
    counts = [sm_counts, l_counts]
    plt.bar(labels, counts, 0.35, label='Symbol Counts')
    plt.title('1/2/3... vs '+str(num_bins)+'+ symbols in GTDB Test Set')
    plt.xlabel('Symbols in each formula')
    plt.ylabel('Counts')
    plt.show()


def viz_char_formula_anns(page_data, pdf_name):
    # Test for a file page comparing original GT with char GT
    orig_ann_file = os.path.join('ScanSSD/ssd/eval/math_gt',pdf_name+'.csv')
    orig_ann = np.genfromtxt(orig_ann_file, delimiter=',')
    q_page = 13
    orig_q_ann = orig_ann[orig_ann[:,0]==q_page][:,1:].astype(
        'int32').reshape((-1,4))
    char_ann = np.array(page_data[q_page][1],dtype=np.int).reshape((-1,4))
    char_ann_xl = np.array(page_data[q_page][0], dtype=np.int).reshape((-1, 4))
    char_ann_m = np.array(page_data[q_page][2], dtype=np.int).reshape((-1, 4))
    char_ann_l = np.array(page_data[q_page][3], dtype=np.int).reshape((-1, 4))
    img_file = os.path.join('ScanSSD/gtdb_data/images',pdf_name,str(q_page+1)+'.png')
    img = cv2.imread(img_file, 1)
    img = draw_rects(img, orig_q_ann, (255,0,0), 10)
    img = draw_rects(img, char_ann, (0,255,0), 3)
    img = draw_rects(img, char_ann_xl, (0, 255, 0), 3)
    img = draw_rects(img, char_ann_m, (0, 255, 0), 3)
    img = draw_rects(img, char_ann_l, (0, 255, 0), 3)
    plt.imshow(img)
    plt.show()


if __name__=='__main__':
    extract_char_data(write=True)