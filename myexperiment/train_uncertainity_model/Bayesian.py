# -*- coding: utf-8 -*- 
# @Time : 2022/9/19 16:06 
# @Author : DirtyBoy 
# @File : Bayesian.py

from experiments.drebin_dataset import data_preprocessing
from core.ensemble.bayesian_ensemble import BayesianEnsemble
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_type', type=str, default="drebin")
    parser.add_argument('-model_type', type=str, default="big")
    args = parser.parse_args()
    model_type = args.model_type
    architecture_type = "dnn"
    feature_type = "drebin"
    model_dir = "drebin_model/"
    if args.train_type == "multi":
        architecture_type = "multimodalitynn"
        feature_type = "multimodality"
        model_dir = "multi_model/"

    if args.model_type == "big":
        mal_folder = "/home/public/rmt/heren/experiment/cl-exp/LHD_apk/apk/train_dnn_model/big_dataset/malware"
        ben_folder = "/home/public/rmt/heren/experiment/cl-exp/LHD_apk/apk/train_dnn_model/big_dataset/benign"
    else:
        mal_folder = "/home/public/rmt/heren/experiment/cl-exp/LHD_apk/apk/train_dnn_model/small_dataset/malware"
        ben_folder = "/home/public/rmt/heren/experiment/cl-exp/LHD_apk/apk/train_dnn_model/small_dataset/benign"

    train_dataset, test_dataset, input_dim = data_preprocessing(feature_type=feature_type, mal_folder=mal_folder,
                                                                ben_folder=ben_folder, model_type=model_type)

    bay_ens20 = BayesianEnsemble(architecture_type=architecture_type,
                                 model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/bayesian/epoch20")

    bay_ens20.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=20)

    MSG20 = "20--" + bay_ens20.evaluate(test_dataset)

    bay_ens30 = BayesianEnsemble(architecture_type=architecture_type,
                                 model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/bayesian/epoch30")

    bay_ens30.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=30)

    MSG30 = "30--" + bay_ens30.evaluate(test_dataset)

    MSG = MSG20 + "\n" + MSG30

    with open(
            '/home/lhd/uncertainity_malware/myexperiment/output/eval_model_log/' + model_dir + model_type + '_bayesian.txt',
            'w') as f:
        f.write(MSG)
