# -*- coding: utf-8 -*- 
# @Time : 2022/9/19 14:58 
# @Author : DirtyBoy 
# @File : Ensemble.py

from experiments.drebin_dataset import data_preprocessing
from core.ensemble.deep_ensemble import DeepEnsemble
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

    d_ensemble1020 = DeepEnsemble(architecture_type=architecture_type, n_members=10,
                                  model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/ensemble/n_member10/epoch20")

    d_ensemble1020.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=20)

    msg1020 = "1020--" + d_ensemble1020.evaluate(test_dataset)

    d_ensemble1030 = DeepEnsemble(architecture_type=architecture_type, n_members=10,
                                  model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/ensemble/n_member10/epoch30")

    d_ensemble1030.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=30)

    msg1030 = "1030--" + d_ensemble1030.evaluate(test_dataset)

    d_ensemble520 = DeepEnsemble(architecture_type=architecture_type, n_members=5,
                                 model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/ensemble/n_member5/epoch20")

    d_ensemble520.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=20)

    msg520 = "520--" + d_ensemble520.evaluate(test_dataset)

    d_ensemble530 = DeepEnsemble(architecture_type=architecture_type, n_members=5,
                                 model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/ensemble/n_member5/epoch30")

    d_ensemble530.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=30)

    msg530 = "530--" + d_ensemble530.evaluate(test_dataset)

    MSG = msg520 + "\n" + msg530 + "\n" + msg1030 + "\n" + msg1020

    with open(
            '/home/lhd/uncertainity_malware/myexperiment/output/eval_model_log/' + model_dir + model_type + '_ensemble.txt',
            'w') as f:
        f.write(MSG)
