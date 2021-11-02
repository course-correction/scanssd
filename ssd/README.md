# ScanSSD: Scanning Single Shot Detector for Math in Document Images

A [PyTorch](http://pytorch.org/) implementation of ScanSSD-XYc [(Link Coming Soon)]() by **Abhisek Dey**. It was developed using SSD implementation by [**Max deGroot**](https://github.com/amdegroot) and is a lighter, more efficient 
alternative to the original [ScanSSD](https://arxiv.org/abs/2003.08005).

Developed using Cuda 11.2 and Pytorch 1.3.0

<img align="right" src=
"https://github.com/maliparag/scanssd/blob/master/images/detailed_math512_arch.png" height = 400/>

&nbsp;
&nbsp;

## Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#code-organization'>Code Organization</a>
- <a href='#training-scanssd'>Training</a>
- <a href='#testing'>Testing</a>
- <a href='#performance'>Performance</a>

&nbsp;
&nbsp;

## Installation
- Refer to Installation instructions in the **main** README.
- To run using the GTDB dataset, Download the dataset by following the instructions on (https://github.com/MaliParag/TFD-ICDAR2019).


## Code Organization
 
SSD model is built in `ssd.py`. Training and testing the SSD is managed in `train.py` and `test.py`. All the training code is in `layers` directory. Hyper-parameters for training and testing can be specified through command line and through `config.py` file inside `data` directory. 

`data` directory also contains `gtdb_iterable.py` data reader that uses sliding windows to generates sub-images of page for training. All the scripts for pooling the sub-image level detections and XY Cuts are in `utils` directory. 

Functions for data augmentation, visualization of bounding boxes and heatmap are also in `utils`. 

## Setting up data for training

If you are not sure how to setup data, use [dir_struct](https://github.com/MaliParag/ScanSSD/blob/master/dir_struct) file. It has the one of the possible directory structure that you can use for setting up data for training and testing. 

To generate .pmath files or .pchar files you can use [this](https://github.com/MaliParag/ScanSSD/blob/master/gtdb/split_annotations_per_page.py) script. 

## Training ScanSSD

- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights [here](https://drive.google.com/file/d/1GqiyZ1TglNW5GrNQfXQ72S8mChhJ4_sD/view?usp=sharing)
- By default, we assume you have downloaded the file in the `ssd/base_weights` dir.
- From the `<root>/src` of the repo, export PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:${PWD}"`
  You may add this path to your `.bashrc` profile to avoid exporting it everytime.
- **Note:** Even if you want to run ScanSSD standalone, you have to run the model from the root directory 
  of the repo (i.e. the pipeline repo). Otherwise, you might run into import issues.

- Example Run command - For training with TFD-ICDARv2 dataset on 2 GPUs. 

```Shell

python3 ScanSSD/ssd/train.py 
--dataset GTDB 
--dataset_root ScanSSD/gtdb_data/ 
--cuda True 
--visdom False
--batch_size 16 
--num_workers 8 
--exp_name ScanSSD_XY_train 
--model_type 512
--suffix _512 
--training_data ScanSSD/file_lists/training_data 
--cfg math_gtdb_512 
--loss_fun ce 
--kernel 1 5 
--padding 0 2 
--neg_mining True 
--stride 0.05
--gpu 0 1
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

## Pre-Trained weights

For quick testing, pre-trained weights are available [here.](https://drive.google.com/file/d/1l81P_QVinPkEjlSYisCfQ5M1X2AFdkbv/view?usp=sharing).
Download and place it in the `ssd/trained_weights` directory.
Alternatively, running the makefile to install the pipeline will automatically download the weights.

## Testing
To test a trained network, from the `src` directory (Make sure you have added this to PYTHONPATH):

```Shell
python3 ScanSSD/ssd/test.py 
--save_folder ScanSSD/ssd/eval/ 
--cuda True 
--dataset_root ScanSSD/gtdb_data/ 
--model_type 512 
--trained_model trained_weights/ssd512GTDB_epoch14.pth 
--cfg math_gtdb_512 
--padding 0 2 
--kernel 1 5 
--batch_size 8  
--log_dir ScanSSD/ssd/logs/Merged_Test_Logs/ 
--test_data ScanSSD/file_lists/testing_data 
--stride 1.0 
--post_process 0 
--conf 0.1 
--gpu 0 1
```

### Visualize results

After prediction have been generated you can overlay them over the original image with the script below. You can also optionally 
overlay the grounds truths as well. Run `python modules/ScanSSD/ssd/gtdb/viz_final_boxes.py --help` for more information.

```shell
python modules/ScanSSD/ssd/utils/viz_final_boxes.py 
--predcsv modules/ScanSSD/ssd/eval/Nested_Test_ssd512GTDB600000/conf_0.3/jones83.csv  
--pagenum 20 
--imgpath modules/ScanSSD/gtdb_data/images/jones83/21.png 
```

## Evaluate 
You can evaluate the detections compared to the ground truth. The csv's should be named after
the pdf document names and should contain the boxes in the format:

`page_num, min_x, min_y, max_x, max_y`

From the `ScanSSD` directory:
```Shell
python3 IOULib/IOUevaluater.py 
--ground_truth ssd/eval/math_gt 
--detections ssd/eval/<exp_name>
```

## Related publications

Dey, Abhisek et al. "ScanSSD-XYc: Faster Detection of Math Formulas", to appear in GREC 2021.

Mali, Parag, et al. “ScanSSD: Scanning Single Shot Detector for Mathematical Formulas in PDF Document Images.” ArXiv:2003.08005 [Cs], Mar. 2020. arXiv.org, http://arxiv.org/abs/2003.08005.

P. S. Mali, ["Scanning Single Shot Detector for Math in Document Images."](https://scholarworks.rit.edu/theses/10210/) Order No. 22622391, Rochester Institute of Technology, Rochester, 2019.

M. Mahdavi, R. Zanibbi, H. Mouchere, and Utpal Garain (2019). [ICDAR 2019 CROHME + TFD: Competition on Recognition of Handwritten Mathematical Expressions and Typeset Formula Detection.](https://www.cs.rit.edu/~rlaz/files/CROHME+TFD%E2%80%932019.pdf) Proc. International Conference on Document Analysis and Recognition, Sydney, Australia (to appear).

## Acknowledgements
- [**Max deGroot**](https://github.com/amdegroot) for providing open-source SSD code
