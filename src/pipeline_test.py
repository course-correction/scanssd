# this is the file to test the pipeline between ScanSSD and SymbolScraper
# the idea: run each simultaneously on a PDF, then output a .tsv file
#   which contains the relevant overlapping data_loaders from the two xml files.


#from jpype import *
import subprocess
import os


def test_jpype():
    startJVM(getDefaultJVMPath(), "-ea")
    java.lang.System.out.println("Calling Java Print from Python using Jpype!")
    shutdownJVM()

def test_call_bash():
    subprocess.call(["modules/SymbolScraper/bin/sscraper", "in_bin", "out_sscraper_bin"])

def process_scanssd(filename):
    filename_without_extension = filename.split(".")[0]
    if not os.path.exists("../../pdf_img_bin/" + filename_without_extension):
        print("WARNING: could not find images for " + filename_without_extension + "PDF, so it was not processed.")
        return # TODO: return something in particular?
    os.system("python3 test.py --dataset_root ../ --trained_model ssd512GTDB60000.pth --visual_threshold 0.25 --cuda True --exp_name " + filename_without_extension + " --save_folder ../../out_scanssd_bin/ --log_dir logs/ --quick_start_data ../pdf_pages_list --model_type 512 --cfg hboxes512 --padding 0 2 --kernel 1 5 --batch_size 8")
#    os.system("python3 gtdb/stitch_patches_pdf.py --data_file ../../pdf_list --exp_name  " + filename_without_extension + "  --math_dir ../../out_scanssd_bin/" + filename_without_extension + "/raw_output/ --stitching_algo equal --algo_threshold 30 --num_workers 8 --postprocess True --eval_dir ../../out_scanssd_bin/ --home_images ../../pdf_img_bin")
#    os.system("python3 visualize_annotations.py --img_dir ../../pdf_img_bin --out_dir ../../out_scanssd_bin/" + filename_without_extension + "/visual_output/ --math_dir ../../out_scanssd_bin/" + filename_without_extension + "/stitched_output/")







process_scanssd("los.pdf")
