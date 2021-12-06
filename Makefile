################################################################
# Makefile for Scanssd
#
# R. Zanibbi, A. Dey, Dec. 1, 2021
#################################################################

SSDW="ssd512GTDB_256_epoch15.pth"

# default: build all, stop on failure
all:
	./install-ssd

# 'make force' - do not stop on failure
force:
	./install-ssd force

test-help:
	./ssdrun --help

# 'make test-example' - run example documents on the default ScanSSD-XYc model
test-example:
	./ssdrun --save_folder src/eval/ --cuda True --dataset_root quick_start_data/ --model_type 512 --trained_model src/trained_weights/ssd512GTDB_256_epoch15.pth --cfg math_gtdb_512 --padding 0 2 --kernel 1 5 --batch_size 2 --log_dir src/logs/ --test_data file_lists/quick_start --stride 1.0 --post_process 0 --conf 0.5 --gpu 0

# 'make train-example' - Run training on the example PDF on the model
train-example:
	./ssdrun-train --dataset GTDB --dataset_root quick_start_data --cuda True --visdom False --batch_size 2 --num_workers 4 --exp_name ScanSSD_XY_train --model_type 512 --suffix _512 --training_data file_lists/quick_start_train --cfg math_gtdb_512 --loss_fun ce --kernel 1 5 --padding 0 2 --neg_mining True --stride 0.05 --gpu 0
# 'make clean' - delete all items from installation script and example.
clean:
	rm -f ./ssdrun
	rm -f ./ssdrun-train
	rm -f ./src/trained_weights/${SSDW}
	rm -rf ./src/eval/SSD/
	rm -rf ./src/weights_ScanSSD_XY_train/
#conda env remove -n scanssd

# 'make clean-example'
clean-outputs:
	rm -rf ./src/eval/SSD/
