# -*- coding: utf-8 -*- 
# @Time : 2024/12/8 9:39 
# @Author : DirtyBoy 
# @File : uncertainty_feature_extractor.py
import pandas as pd
import numpy as np
import os, argparse
from metrics_utils import *
from utils import metrics_dict, read_joblib


def main_(train_data_type, feature_type, test_data_type):
    save_path = os.path.join('../Training/config',
                             f'databases_{feature_type}_{test_data_type}.conf')

    print('load filename and label from ' + save_path)
    data_filenames, gt_labels = read_joblib(save_path)

    vanilla_prob = np.squeeze(
        np.load(f'../Training/output/{feature_type}/{train_data_type}_vanilla_{test_data_type}.npy'))

    data = {
        "apknames": data_filenames,
        "gt_labels": gt_labels,
        "vanilla+prob": vanilla_prob
    }
    df = pd.DataFrame(data)

    columns_names = []
    metrics_list = []
    for model_type in uc_model_types:
        model_output = np.load(
            f'../Training/output/{feature_type}/{train_data_type}_{model_type}_{test_data_type}.npy')
        for metrics in metrics_types:
            columns_names.append(f'{model_type}+{metrics}')
            if model_type == 'wensemble' and (metrics == 'kld' or metrics == 'std' or metrics == 'entropy'):
                w = np.load(
                    f'../Training/output/{feature_type}/{train_data_type}_{model_type}_{test_data_type}_weights.npy')
                metric_values = np.array([metrics_dict[metrics](prob_, w=w) for prob_ in model_output])
            else:
                metric_values = np.array([metrics_dict[metrics](prob_) for prob_ in model_output])
            metrics_list.append(metric_values)

    for col_name, col_values in zip(columns_names, metrics_list):
        df[col_name] = col_values
    df.to_csv(f"{csv_save_path}/uc_metrics_{feature_type}_{train_data_type}_{test_data_type}.csv", index=False)


uc_model_types = ['epoch_ensemble', 'bayesian', 'mc_dropout', 'ensemble', 'wensemble']
metrics_types = ['entropy', 'kld', 'std', 'max_max2', 'min2_min', 'max_min', 'mean_med']
test_data_types = ['earliest', '2013', '2016', 'latest']
if __name__ == '__main__':

    csv_save_path = 'uncertain_feature'
    if not os.path.isdir(csv_save_path):
        os.makedirs(csv_save_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('-test_data_type', '-tdt', type=str, default="latest")
    parser.add_argument('-train_data_type', '-dt', type=str, default="base")
    parser.add_argument('-feature_type', '-ft', type=str, default="drebin")
    args = parser.parse_args()
    feature_type = args.feature_type
    test_data_type = args.test_data_type
    train_data_type = args.train_data_type

    main_(train_data_type, feature_type, test_data_type)
