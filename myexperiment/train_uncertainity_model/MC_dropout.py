# -*- coding: utf-8 -*- 

# @File : MC_dropout.py


from experiments.drebin_dataset import data_preprocessing
from core.ensemble.mc_dropout import MCDropout
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

   
    mc_dp30 = MCDropout(architecture_type=architecture_type,
                        model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/mc_dropout/epoch30")

    mc_dp30.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=30)

    MSG30 = "30--" + mc_dp30.evaluate(test_dataset)


    with open(
            '/home/lhd/uncertainity_malware/myexperiment/output/eval_model_log/' + model_dir + model_type + '_mcdropout.txt',
            'w') as f:
        f.write(MSG)
