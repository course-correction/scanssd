import logging
import torch
import os
import shutil
import cv2
import tqdm
import numpy as np
import time

from src.utils.xy_splitting_numpy import create_splits
from src.utils.stitch_patches_pdf import preprocess_math_regions


# Writes out CSV files, one per document / directory of images
# CSVs are written per document, with the following format for each detection region:
#
#   Page#, minX, minY, maxX, maxY  (all floats, 2 decimal precision)
#
# page numbers start from 0 (e.g., for a six page documents, page nums from 0.00 to 5.00.

def process_page(args, page_data, doc_count, file_name, return_data=False):
    start_time = time.time()

    logging.debug(
        '\n  XY-cut overlapping detection merge ({} docs(s), {} page(s))'.format(doc_count, len(page_data.keys())))
    # logging.debug(">>>>> " + str(page_data.keys()))
    # logging.debug(">> arg.exp_name: " + args.exp_name)

    # Check if detections for the run already exist, then delete the folder to recompute
    if not return_data:
        save_folder = os.path.join(args.save_folder, args.exp_name)
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)

    full_csv = None

    for page in tqdm.tqdm(page_data.keys(), desc="  Processing"):

        # Process detection regions, if any were found on the current page.
        boxes = page_data[page]
        if len(boxes):
            # extract needed information
            if return_data:
                page_num = float(page.split('_')[0])
            else:
                pdf_name = page.split("/")[0]
                page_num_with_conf = page.split("/")[1]
                page_num = float(page_num_with_conf.split('_')[0])
                conf = float(page_num_with_conf.split('_')[1])
                # print (">> pdf_name: " + pdf_name)

                # Create CSV file path, and CSV directory if missing.
                math_csv_path = os.path.join(args.save_folder,
                                             args.exp_name,  # file_name,
                                             "conf_" + str(conf), pdf_name + ".csv")
                # print("  >> Saving: " + str(math_csv_path) )
                if not os.path.exists(os.path.dirname(math_csv_path)):
                    os.makedirs(os.path.dirname(math_csv_path))

            new_b = boxes[0][0]
            new_sc = boxes[0][1]
            for idx, box_c in enumerate(boxes[1:]):
                new_sc = np.concatenate((new_sc, box_c[1]))
                new_b = np.concatenate((new_b, box_c[0]), axis=0)

            # Apply X-Y cutting
            final_boxes = []
            splits = create_splits(new_b, new_sc)
            for split in splits:
                grp_boxes = split[0]
                x_min, x_max = np.min(grp_boxes[:, 0]), np.max(grp_boxes[:, 2])
                y_min, y_max = np.min(grp_boxes[:, 1]), np.max(grp_boxes[:, 3])
                final_boxes.append([x_min, y_min, x_max, y_max])

            # Apply post-processing if requested.
            if args.post_process:
                img_path = os.path.join(args.dataset_root, 'images', img_id + '.png')
                img = cv2.imread(img_path, 1)
                final_boxes = preprocess_math_regions(final_boxes, img)

            final_boxes = np.array(final_boxes, dtype=np.float).reshape((-1, 4))
            final_math = np.concatenate((np.full((len(final_boxes), 1), page_num - 1, dtype=np.float),
                                         final_boxes), axis=1)

            # RZ Note: regions are not sorted by page or location on output.

            # Save final regions to CSV file.
            if return_data:
                if full_csv is None:
                    full_csv = final_math
                else:
                    full_csv = np.concatenate((full_csv, final_math), axis=1)
                #full_csv = full_csv + to_csv(final_math)
            else:
                math_file = open(math_csv_path, 'a')
                np.savetxt(math_file, final_math, fmt='%.2f', delimiter=',')
                math_file.close()

    logging.debug("  Merge time: {:.2f} seconds".format(time.time() - start_time))
    if return_data:
        return full_csv
    return 0


def to_csv_line(line):
    return ','.join([str(num) for num in line])


def to_csv(arr):
    csv = ''
    for line in arr:
        csv = csv + to_csv_line(line) + '\n'
    return csv
