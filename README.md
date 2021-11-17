# ScanSSD: Scanning Single Shot Detector for Math in Document Images

A [PyTorch](http://pytorch.org/) implementation of ScanSSD-XYc [(Link)](https://www.cs.rit.edu/~rlaz/files/ScanSSDv2.pdf) by **Abhisek Dey**. It was developed using SSD implementation by [**Max deGroot**](https://github.com/amdegroot) and is a lighter, more efficient 
alternative to the original [ScanSSD](https://arxiv.org/abs/2003.08005).

Developed using Cuda 11.2 and Pytorch 1.3.0

&nbsp;
&nbsp;

## Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#code-organization'>Code Organization</a>
- <a href='#training-scanssd'>Training</a>
- <a href='#testing'>Testing</a>
- <a href='#performance'>Performance</a>

## Installation
First make sure you have Anaconda3 installed on your system. Then run the following commands
```zsh
$ conda create -n scanssd python=3.6.9
$ conda activate scanssd
(scanssd) $ pip install -r requirements.txt
```
To run using the GTDB dataset, Download the dataset by following the instructions on (https://github.com/MaliParag/TFD-ICDAR2019).
## Code Organization

SSD model is built in `ssd.py`. Training and testing the SSD is managed in `train.py` and `test.py`. All the training code is in `layers` directory. Hyper-parameters for training and testing can be specified through command line and through `config.py` file inside `data` directory. 

`data` directory also contains `gtdb_iterable.py` data reader that uses sliding windows to generates sub-images of page for training. All the scripts for pooling the sub-image level detections and XY Cuts are in `utils` directory. 

Functions for data augmentation, visualization of bounding boxes and heatmap are also in `utils`. 

## Setting up data for training

If you are not sure how to setup data, use [dir_struct directory](https://github.com/MaliParag/ScanSSD/blob/master/dir_struct) file. It has the one of the possible directory structure that you can use for setting up data for training and testing. 

To generate .pmath files (csv files containing only numbers for bounding-box coordinates, 1 per page) or .pchar (same as .pmath but contains character-based box coordinates) files you can use [this](https://github.com/MaliParag/ScanSSD/blob/master/gtdb/split_annotations_per_page.py) script. 

## Training ScanSSD

- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights [here](https://drive.google.com/file/d/1GqiyZ1TglNW5GrNQfXQ72S8mChhJ4_sD/view?usp=sharing)
- By default, we assume you have downloaded the file in the `ssd/base_weights` dir.
- From the `<root>` of this repo, export PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:${PWD}"`
  You may add this path to your `.bashrc` profile to avoid exporting it everytime.

- Example Run command - For training with TFD-ICDARv2 dataset on GPU 0. 

```Shell
python3 src/train.py \
--dataset GTDB \
--dataset_root quick_start_data \
--cuda True \
--visdom False \
--batch_size 4 \
--num_workers 4 \
--exp_name ScanSSD_XY_train \
--model_type 512 \
--suffix _512 \
--training_data file_lists/quick_start_train \
--cfg math_gtdb_512 \
--loss_fun ce \
--kernel 1 5 \
--padding 0 2 \
--neg_mining True \
--stride 0.05 \
--gpu 0
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

## Pre-Trained weights

For quick testing, pre-trained weights are available [here.](https://drive.google.com/file/d/1CG8Z6R-BS9SL2ntFo8ruJhWbg8yaIuik/view?usp=sharing).
Download and place it in the `src/trained_weights` directory.
Alternatively, running the makefile to install the pipeline will automatically download the weights.

## Testing
To test a trained network (Make sure you have added this to PYTHONPATH):

```Shell
python3 src/test.py \
--save_folder src/eval/ \
--cuda True \
--dataset_root quick_start_data/ \
--model_type 512 \
--trained_model src/trained_weights/ssd512GTDB_256_epoch15.pth \
--cfg math_gtdb_512 \
--padding 0 2 \
--kernel 1 5 \
--batch_size 8 \
--log_dir src/logs/ \
--quick_start_data file_lists/quick_start_data \
--stride 1.0 \
--post_process 0 \
--conf 0.1 \
--gpu 0
```

### Visualize results

After prediction have been generated you can overlay them over the original image with the script below. You can also optionally 
overlay the grounds truths as well. Run `python src/gtdb/viz_final_boxes.py --help` for more information. The script will generate an output image
containing the predictions (in <span style="color:green">green</span>) boxes overlaid with the actual ground-truth boxes
(in <span style="color:red">red</span>) boxes. The image will be saved in the root directory (from where the program was called).

```shell
python src/utils/viz_final_boxes.py \
--predcsv src/eval/SSD/conf_0.1/Emden76.csv \
--pagenum 1 \
--imgpath quick_start_data/images/Emden76/2.png
```

## Evaluate 
You can evaluate the detections compared to the ground truth. The csv's should be named after
the pdf document names and should contain the boxes in the format:

`page_num, min_x, min_y, max_x, max_y`

From the root directory:
```Shell
python3 IOU_lib/IOUevaluater.py \
--ground_truth quick_start_data/gt/ \
--detections src/eval/SSD/conf_0.1/ \
--exp_name ScanSSD_XY
```

Where:

`--ground_truth` - GT CSV files for the documents tested

`--detections` - Detection folder for the epoch and confidence level

`--exp_name` - The detection result will be saved in a csv in `ssd/metrics/<exp_name>/<weight_name>.csv`

### Evaluate on multiple weights and multiple confidence levels (When testing is run on multiple confidence levels)
You can also use the same script to evaluate multiple epoch results (with multiple confidence threshold levels). Run the script on the epochs you want to evaluate keeping the directory as:

**Note:** Make sure that your detection folder structure follows the following structure:

`test_save_folder/ssd512GTDB_epoch<number>/conf_<level>/<doc_names>.csv`

The entire directory structure should look like:

```
<test_save_folder>
  ssd512GTDB_epoch1
    conf_0.1
      <CSVs>
    conf_0.25
      <CSVs>
    conf_0.5
      .
      .
      .
  ssd512GTDB_epoch2
    .
    .
    .   
```
For using IOUevaluater on a folder containing prediction csv files for different epochs at different confidence levels, use the following format for each epoch:
```Shell
python IOU_lib/IOUevaluater.py \
--ground_truth quick_start_data/gt \
--det_folder ssd/eval/<detection_folder> \
--exp_name <Name of Exp Folder>
```

Where:

`--det_folder` - The detection folder for an epoch. From the above structure it would be 
**ssd512GTDB_epoch1** or **ssd512GTDB_epoch2** and so on 

**Note:** Keep the `exp_name` same for all epochs for the same testing run.

You have to run each weight file you want to evaluate separately and keep the same `exp_name` for the same set of weights. It can handle one or many confidence levels.
For each epoch your experiment folder will have a corresponding output csv file named `ssd512GTDB_epoch<number>_overall.csv`.
The experiment folder will be in the `ScanSSD/ssd/metrics` directory

### Determining best weight file to use
After you have obtained weight files you can choose a metric out of 
```python
metrics = ['F_0.01', 'P_0.01', 'R_0.01', 'F_0.25', 'P_0.25', 'R_0.25', 'F_0.5', 'P_0.5', 'R_0.5', 'F_0.75', 'P_0.75', 'R_0.75', 'F_1.0', 'P_1.0', 'R_1.0']
```
Where `R_0.5` would give the best weight file and recall value at 50% IoU, `F_0.75` would give the best F-score at 75% IoU and so on.

You can use the following:
```shell
python IOU_lib/get_best.py \
--exp_folder <Path to the CSV outputs containing metrics> \
--metrics <One or many metrics to compare>
```

**Note:** Metrics can be one or many. For eg: `--metrics R_0.5 R_0.75` or `--metrics F_0.5`. The program will give you the best weight to use for each metric.

## Related publications

Dey, Abhisek et al. "ScanSSD-XYc: Faster Detection of Math Formulas", GREC 2021 [link](https://www.cs.rit.edu/~rlaz/files/ScanSSDv2.pdf).

Mali, Parag, et al. “ScanSSD: Scanning Single Shot Detector for Mathematical Formulas in PDF Document Images.” ArXiv:2003.08005 [Cs], Mar. 2020. arXiv.org, http://arxiv.org/abs/2003.08005.

P. S. Mali, ["Scanning Single Shot Detector for Math in Document Images."](https://scholarworks.rit.edu/theses/10210/) Order No. 22622391, Rochester Institute of Technology, Rochester, 2019.

M. Mahdavi, R. Zanibbi, H. Mouchere, and Utpal Garain (2019). [ICDAR 2019 CROHME + TFD: Competition on Recognition of Handwritten Mathematical Expressions and Typeset Formula Detection.](https://www.cs.rit.edu/~rlaz/files/CROHME+TFD%E2%80%932019.pdf) Proc. International Conference on Document Analysis and Recognition, Sydney, Australia (to appear).

## Acknowledgements
- [**Max deGroot**](https://github.com/amdegroot) for providing open-source SSD code
