# -*- coding: utf-8 -*- 
# @Time : 2022/9/19 16:26 
# @Author : DirtyBoy 
# @File : Wensemble.py
import argparse
from experiments.drebin_dataset import data_preprocessing
from core.ensemble.deep_ensemble import WeightedDeepEnsemble

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

   
    w_ensemble1030 = WeightedDeepEnsemble(architecture_type=architecture_type, n_members=10,
                                          model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/wensemble/n_member10/epoch30")

    w_ensemble1030.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=30)

    msg1030 = "1030--" + w_ensemble1030.evaluate(test_dataset)

    MSG = msg1030 

    with open(
            '/home/lhd/uncertainity_malware/myexperiment/output/eval_model_log/' + model_dir + model_type + '_wensemble.txt',
            'w') as f:
        f.write(MSG)
