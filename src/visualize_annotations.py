import csv
import cv2
import os
import sys
import argparse
import logging
import time

# RZ HACK: Global for page resizing for visualization.
SCALE=0.5

"""
    This module annotates math regions and character information on input document images.
    Usage: "python3 visualize_annotations.py image_dir math_dir output_dir"
"""

class Doc():
    '''
        Object to store character and math region information.
    '''
    def __init__(self):
        self.sheet_math_map = {}
        self.sheet_char_map = {}
        self.sheet_cc_map = {}
        self.sheet_word_map = {}

    def add_math_region(self, row):
        # ASSUMES we are reading region CSV file.
        '''
        add math bouding box
        :param row:
        :return: None
        '''
        if float(row[0]) not in self.sheet_math_map:
            self.sheet_math_map[float(row[0])] = []

        self.sheet_math_map[float(row[0])].append((float(row[1]), float(row[2]), float(row[3]), float(row[4])))

    def add_char_region(self, row):
        '''
        add character bouding box
        :param row:
        :return: None
        '''
        if float(row[0]) not in self.sheet_char_map:
            self.sheet_char_map[float(row[0])] = []

        self.sheet_char_map[float(row[0])].append((float(row[3]), float(row[4]), float(row[5]), float(row[6])))

    # Adding ability to read TSV output directly.
    def add_tsv_region(self, row, currentPage, char_count, formula_count, cc_count, word_count ):
        '''
        add formula and character bounding boxes.
        :param row:
        :return: None
        '''
        # Formula entry
        if row[0] == "P":
            # New format requires us to update pages as we read them.
            currentPage = int(row[1]) - 1 

        if row[0] == "FR":
            if ( currentPage ) not in self.sheet_math_map:
                self.sheet_math_map[ currentPage ] = []
            self.sheet_math_map[ currentPage ].append((float(row[2]), float(row[3]), float(row[4]), float(row[5])))

            return ( currentPage,  char_count, formula_count + 1, cc_count, word_count )
        
        # Symbol (character) entry -- page # depends on previous formula entry.
        # Need to skip characters in regions!
        elif row[0] == "S":
            if ( currentPage ) not in self.sheet_char_map:
                self.sheet_char_map[ currentPage ] = []
            self.sheet_char_map[ currentPage ].append((float(row[3]), float(row[4]), float(row[5]), float(row[6])))

            return ( currentPage, char_count + 1, formula_count, cc_count, word_count )

        # RZ: adding connected components.
        elif row[0] == "c":
            if ( currentPage ) not in self.sheet_cc_map:
                self.sheet_cc_map[ currentPage ] = []
            self.sheet_cc_map[ currentPage ].append((float(row[3]), float(row[4]), float(row[5]), float(row[6])))

            return ( currentPage, char_count, formula_count, cc_count + 1, word_count)

        # RZ: Adding words
        elif row[0] == "W":
            if ( currentPage ) not in self.sheet_word_map:
                self.sheet_word_map[ currentPage ] = []
            self.sheet_word_map[ currentPage ].append((float(row[3]), float(row[4]), float(row[5]), float(row[6])))

            return ( currentPage, char_count, formula_count, cc_count, word_count + 1 )
        else:
            # Skip line.
            return ( currentPage, char_count, formula_count, cc_count, word_count )



    def read_file(self, filename, is_math = False, delim=","):
        '''
        Read .math file and parse contents into Doc() object
        :param filename: ".math" annotations filepath
        :param doc: Doc object to store math regions (BBox)
        :return: None
        '''
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = delim )
            formula_count = 0
            char_count = 0
            cc_count = 0
            word_count = 0
            current_page = 0

            for row in csv_reader:
                if is_math:
                    self.add_math_region(row)
                    formula_count += 1
                else:
                    if delim == ",":
                        self.add_char_region(row)

                    # TSV files, character page is determined by previous formula entry
                    elif delim == "\t":
                        ( current_page, char_count, formula_count, cc_count, word_count ) = self.add_tsv_region(row, current_page, char_count, formula_count, cc_count, word_count )

            print( "  * " + str(os.path.abspath(filename)) + "    \n    [ %d Formulas, %d PDF Words, %d PDF Chars, %d CCs  ]"%(formula_count, word_count, char_count, cc_count ))


def draw_rectangle_cv( img_file, math_bboxes, output_path, char_bboxes, cc_bboxes, word_bboxes):
    '''
    Draw math bounding boxes on image file and save resulting image in output path
    :param img_file: input document image filepath
    :param math_bboxes: List of math bounding boxes
    :param output_path: output directory to store annotated document images
    :return: None
    '''

    img = cv2.imread(img_file)
    if math_bboxes:
        for math_bbox in math_bboxes:
            math_left = math_bbox[0]
            math_top = math_bbox[1]
            math_right = math_bbox[2]
            math_bottom = math_bbox[3]
            cv2.rectangle(img, (int(math_left), int(math_top)), (int(math_right), int(math_bottom)), (255, 0, 0), 4)

    if word_bboxes:
        for word_bbox in word_bboxes:
            cc_left = word_bbox[0]
            cc_top = word_bbox[1]
            cc_right = word_bbox[2]
            cc_bottom = word_bbox[3]
            cv2.rectangle(img, (int(cc_left), int(cc_top)), (int(cc_right), int(cc_bottom)), (180, 0, 255), 3)

    if char_bboxes:
        for char_bbox in char_bboxes:
            char_left = char_bbox[0]
            char_top = char_bbox[1]
            char_right = char_bbox[2]
            char_bottom = char_bbox[3]
            cv2.rectangle(img, (int(char_left), int(char_top)), (int(char_right), int(char_bottom)), (0, 255, 140), 2)

    if cc_bboxes:
        for cc_bbox in cc_bboxes:
            cc_left = cc_bbox[0]
            cc_top = cc_bbox[1]
            cc_right = cc_bbox[2]
            cc_bottom = cc_bbox[3]
            cv2.rectangle(img, (int(cc_left), int(cc_top)), (int(cc_right), int(cc_bottom)), (0, 0, 255), 2)



    # RZ: Reduce image sizes - currently ~5.3MB each!
    #origWidth = img.shape[1]
    #origHeight = img.height[0]
    width = round( img.shape[1] * SCALE )
    height = round( img.shape[0] * SCALE )
    dim = (width,height)
    resized = cv2.resize( img, dim, interpolation = cv2.INTER_CUBIC )

    # Write file to disk, in output directory.
    cv2.imwrite(os.path.join(output_path, img_file.split(os.sep)[-1]), resized )


# RZ: adding for TSV.
def annotate_tsv( img_dir, tsv_file, out_dir ):
    doc = Doc()
    doc.read_file( tsv_file, False, '\t' )

    # Get raster page images.
    img_files = {}
    for dirName, subdirList, fileList in os.walk(img_dir):
        for img_filename in fileList:
            if img_filename.endswith(".png"):
                img_files[int(img_filename.split(".png")[0]) - 1] = img_filename
        break

    img_name = img_dir.split(os.sep)[-1]
    output_path = os.path.join(out_dir, img_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create annotated images (one image per page)
    for i in sorted(img_files.keys()):
        img = cv2.imread(os.path.join(img_dir, img_files[i]))
        cv2.imwrite(os.path.join(output_path, img_files[i]), img)
        if not i in doc.sheet_math_map.keys():
            math = None
        else:
            math = doc.sheet_math_map[i]

        if not i in doc.sheet_char_map.keys():
            chars = None
        else:
            chars = doc.sheet_char_map[i]

        if not i in doc.sheet_cc_map.keys():
            ccs = None
        else:
            ccs = doc.sheet_cc_map[i]

        if not i in doc.sheet_word_map.keys():
            words = None
        else:
            words = doc.sheet_word_map[i]

        draw_rectangle_cv( os.path.join(img_dir, img_files[i]), \
                math, output_path, chars, ccs, words)


def annotate_each_file(img_dir, math_file, char_file, out_dir):
    '''
    Annotate
    :param img_dir: image directory containing all images of a single PDF file
    :param math_file: .math file containing math regions bounding boxes info corresponding to PDF file
    :param out_dir: output path to save annotated document images
    :return:
    '''
    doc = Doc()
    if math_file:
        doc.read_file(math_file, True)
    if char_file:
        # RZ: Adding switch to different behavior for TSV
        if char_file.endswith('.tsv'):
            doc.read_file(char_file, False, '\t')
            # Indicate for logic below that we have a math and char file.
            math_file = char_file
        else:
            doc.read_file(char_file)
    
    # Get raster page images.
    img_files = {}
    for dirName, subdirList, fileList in os.walk(img_dir):
        for img_filename in fileList:
            if img_filename.endswith(".png"):
                img_files[int(img_filename.split(".png")[0]) - 1] = img_filename
        break

    img_name = img_dir.split(os.sep)[-1]
    output_path = os.path.join(out_dir, img_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # RZ: This logic can likely be greatly reduced here, it is awkward.
    for i in sorted(img_files.keys()):
        # print(char_file, math_file)
        if not char_file and math_file:
            if i not in doc.sheet_math_map.keys():
                img = cv2.imread(os.path.join(img_dir, img_files[i]))
                cv2.imwrite(os.path.join(output_path, img_files[i]), img)
                continue
            draw_rectangle_cv(os.path.join(img_dir, img_files[i]), doc.sheet_math_map[i], output_path, None, None, None)
        elif char_file and not math_file:
            if i not in doc.sheet_char_map.keys():
                img = cv2.imread(os.path.join(img_dir, img_files[i]))
                cv2.imwrite(os.path.join(output_path, img_files[i]), img)
                continue
            draw_rectangle_cv(os.path.join(img_dir, img_files[i]), None, output_path, doc.sheet_char_map[i], None, None)
        elif char_file and math_file:
            if (i not in doc.sheet_math_map.keys()) and (i not in doc.sheet_char_map.keys()):
                img = cv2.imread(os.path.join(img_dir, img_files[i]))
                cv2.imwrite(os.path.join(output_path, img_files[i]), img)
                continue
            elif (i in doc.sheet_math_map.keys()) and (i in doc.sheet_char_map.keys()):
                draw_rectangle_cv(os.path.join(img_dir, img_files[i]), doc.sheet_math_map[i], output_path, doc.sheet_char_map[i], None, None )
            elif (i in doc.sheet_math_map.keys()) and (i not in doc.sheet_char_map.keys()):
                draw_rectangle_cv(os.path.join(img_dir, img_files[i]), doc.sheet_math_map[i], output_path, None, None, None)
            elif (i not in doc.sheet_math_map.keys()) and (i in doc.sheet_char_map.keys()):
                draw_rectangle_cv(os.path.join(img_dir, img_files[i]), None, output_path, doc.sheet_char_map[i], None, None)



if __name__ == '__main__':

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Directory of pdf images")
    parser.add_argument("--out_dir", type=str, required=True, help="Destination to store annotated images")
    parser.add_argument("--math_dir", type=str, required=False, help="Directory of math annotations")
    parser.add_argument("--char_dir", type=str, required=False, help="Directory of char annotations")
    parser.add_argument("--tsv_dir", type=str, required=False, help="Directory of TSV files")
    parser.add_argument("--suffix", type=str, required=False, help="Suffix for TSV file names")
    args = parser.parse_args()

    # Announce what is happening.
    print("\n[ Rendering Image Regions ]")
    if args.math_dir != None:
        print("  Region dir: " + os.path.abspath(args.math_dir))
    
    if args.char_dir != None:
        print("  Char dir: " + os.path.abspath(args.char_dir))

    if args.tsv_dir != None:
        print("  TSV dir: " +  os.path.abspath(args.tsv_dir))

    print("  Image dir: " + os.path.abspath(args.img_dir))
    print("  Output dir: " + os.path.abspath(args.out_dir))


    '''
        store image directory paths
    '''
    img_sub_dirs = {}
    for dirName, subdirList, fileList in os.walk(args.img_dir):
        for subdir in subdirList:
            # RZ: Skip empty directories (e.g., .md directories 
            # while testing early part of the pipeline)
            if subdir not in img_sub_dirs:
                img_sub_dirs[subdir] = os.path.join(args.img_dir,subdir)
        break

    '''
        store math files and char files paths
    '''
    math_files = {}
    if args.math_dir:
        for dirName, subdirList, fileList in os.walk(args.math_dir):
            for filename in fileList:
                if filename.endswith(".math"):
                    math_files[filename.split(".math")[0]] = os.path.join(args.math_dir, filename)
                if filename.endswith(".csv"):
                    math_files[filename.split(".csv")[0]] = os.path.join(args.math_dir, filename)
            break

    # RZ: store TSV files separately.
    char_files = {}
    tsv_files = {}
    if args.char_dir:
        for dirName, subdirList, fileList in os.walk(args.char_dir):
            for filename in fileList:
                if filename.endswith(".char"):
                    char_files[filename.split(".char")[0]] = os.path.join(args.char_dir, filename)
                #if filename.endswith(".csv"):
                #    char_files[filename.split(".csv")[0]] = os.path.join(args.char_dir, filename)
            break

    # RZ modified walk for TSV 
    suffix = args.suffix
    if args.suffix == None:
        suffix = ''

    fileEnd = suffix + ".tsv"
    if args.tsv_dir:
        for ( dirName, subdirList, fileList ) in os.walk(args.tsv_dir):
            for filename in fileList:
                if filename.endswith(fileEnd):
                    tsv_files[filename.split(fileEnd)[0]] = \
                            os.path.join(args.tsv_dir, filename ) 

    for key in img_sub_dirs.keys():
        #print("  Dir: " + img_sub_dirs[key])
        #print(math_files[key])
        # print(char_files[key])
        if key in tsv_files:
            annotate_tsv( img_sub_dirs[key], tsv_files[key], args.out_dir )

        elif (key in char_files) and (key in math_files):
            annotate_each_file(img_sub_dirs[key], math_files[key], char_files[key], args.out_dir)
        elif (key in char_files) and (key not in math_files) :
            annotate_each_file(img_sub_dirs[key], None, char_files[key], args.out_dir)

        elif (key not in char_files) and (key in math_files):
            annotate_each_file(img_sub_dirs[key], math_files[key], None, args.out_dir)

        else:
            print("  >> **Warning: " + key + " not present in annotations")

    print("  Rendering time: {:.2f}".format( time.time() - start ) + " seconds")
