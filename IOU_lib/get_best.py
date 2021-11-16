import os
import glob
import numpy as np
import argparse

def compute_best(eval_dir, metrics):
    avail_met = ['F_0.01', 'P_0.01', 'R_0.01', 'F_0.25', 'P_0.25', 'R_0.25',
                    'F_0.5', 'P_0.5', 'R_0.5', 'F_0.75', 'P_0.75', 'R_0.75',
                    'F_1.0', 'P_1.0', 'R_1.0']
    comp_cols = []
    for metric in metrics:
        comp_cols.append(avail_met.index(metric)+1)

    files = glob.glob(os.path.join(eval_dir, '*.csv'))
    file_names = [file.split('/')[-1].split('_')[1] for file in files]

    best_val = [0]*len(comp_cols)
    best_weight = ['']*len(comp_cols)
    best_conf = [0.5]*len(comp_cols)

    for idx,file in enumerate(files):
        data = np.genfromtxt(file, skip_header=2, delimiter=',')
        for conf_level in data:
            for idx_c, col in enumerate(comp_cols):
                if best_val[idx_c] < conf_level[col]:
                    best_val[idx_c] = conf_level[col]
                    best_weight[idx_c] = file_names[idx]
                    best_conf[idx_c] = conf_level[0]

    # Print the results
    for i in range(len(comp_cols)):
        print(f'For metric {avail_met[comp_cols[i]-1]}, the metrics are:')
        print(f'The best weight file found is of epoch: {best_weight[i]}')
        print(f'The best average for the metric chosen was: {best_val[i]}')
        print(f'The best confidence threshold was found to be: {best_conf[i]}')


if __name__ == '__main__':
   # compute_best('ssd/metrics/chem_256_S1_HC', comp_cols=[12])
    parser = argparse.ArgumentParser(description="Get the best weight file epoch "
                                                "from a set of CSV outputs")
    parser.add_argument('--exp_folder', required=True, type=str,
                        help='The path to the folder containing the csv metrics '
                            'for all epochs')
    parser.add_argument('--metrics', required=True, nargs='+', type=str,
                        help='Choose from [F_0.01, P_0.01, R_0.01, F_0.25, P_0.25,'
                             ' R_0.25, F_0.5, P_0.5, R_0.5, F_0.75, P_0.75, R_0.75, '
                             'F_1.0, P_1.0, R_1.0]')
    args = parser.parse_args()
    compute_best(args.exp_folder, args.metrics)
