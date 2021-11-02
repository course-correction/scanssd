import numpy as np
import os
import glob
from tqdm import tqdm
from src.utils.msFileIO import region_intersection
from read_char_gt import write_char_data, count_chars, compare_anns, plot_hist, viz_char_formula_anns

CHAR_CT_DICT = {p: 0 for p in range(20)}
MATCHES_DICT = {p: 0 for p in range(5)}


def get_intersection(write=False, match=False, count=False):
    char_files_folder = 'src/ScanSSD/gtdb_data/GT_char_no_label'
    char_files = glob.glob(os.path.join(char_files_folder,'*.char'))
    formula_files = 'src/ScanSSD/ssd/eval/math_gt'
    annotation_folder = 'src/ScanSSD/ssd/eval/'

    for char_file in tqdm(char_files):
        page_data = {}
        pdf_name = char_file.split('/')[-1].replace('.char', '')
        char_ann = np.genfromtxt(char_file, delimiter=',').astype('int32')
        form_ann_file = os.path.join(formula_files,pdf_name+'.csv')
        form_ann = np.genfromtxt(form_ann_file, delimiter=',').astype('int32')
        pages = np.unique(form_ann[:,0]).astype('int32')

        char_ann = char_ann[:,[0,2,3,4,5,6,7]]
        form_dict = {p:list(form_ann[form_ann[:,0] == p][:,1:]) for p in pages}
        char_dict = {p:list(char_ann[char_ann[:,0] == p][:,1:]) for p in pages}

        objTable = region_intersection(form_dict,0,char_dict)

        for p in pages:
            sym_data = {num_sym: [] for num_sym in range(20)}
            for form_box_info in objTable[p]:
                if len(form_box_info[1]):
                    num_syms = len(form_box_info[1])
                    syms_form_boxes = np.array(form_box_info[1])[:,:4].reshape((-1,4))
                    x_min, x_max = np.min(syms_form_boxes[:, 0]), np.max(syms_form_boxes[:, 2])
                    y_min, y_max = np.min(syms_form_boxes[:, 1]), np.max(syms_form_boxes[:, 3])
                    if num_syms >= 20:
                        sym_data[0].append([x_min, y_min, x_max, y_max])
                    else:
                        sym_data[num_syms].append([x_min, y_min, x_max, y_max])
            page_data[p] = sym_data

        if write:
            write_char_data(page_data, annotation_folder, pdf_name)
        if count:
            count_chars(page_data, CHAR_CT_DICT)
        if match:
            compare_anns(page_data, pdf_name, MATCHES_DICT, form_dict)
        # if pdf_name=='Kazhdan79':
        #     viz_char_formula_anns(page_data, pdf_name)

    if count:
        print(CHAR_CT_DICT)
        plot_hist(CHAR_CT_DICT, 20)
    if match:
        print(MATCHES_DICT)

if __name__ == '__main__':
    get_intersection(write=False, match=False, count=True)
