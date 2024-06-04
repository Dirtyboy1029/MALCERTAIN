# -*- coding: utf-8 -*- 
# @Time : 2022/9/19 21:14 
# @Author : DirtyBoy 
# @File : mcdropout_evaluate.py

from experiments.oos import data_preprocessing as oos_data_preprocessing
from experiments.adv import data_preprocessing
from core.ensemble.mc_dropout import MCDropout
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_type', type=str, default="drebin")
    parser.add_argument('-model_type', type=str, default="big")
    parser.add_argument('-data_type', type=str, default="adv")
    args = parser.parse_args()

    model_type = args.model_type
    architecture_type = "dnn"
    feature_type = "drebin"
    model_dir = "drebin_model/"
    if args.train_type == "opcode":
        architecture_type = "droidectc"
        feature_type = "opcodeseq"
        model_dir = "opcode_model/"

    mc_20 = MCDropout(architecture_type=architecture_type,
                      model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/mc_dropout/epoch20")
    mc_30 = MCDropout(architecture_type=architecture_type,
                      model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/mc_dropout/epoch30")

    if args.data_type == "adv":
        prist_data, adv_data, prist_y, adv_y, input_dim = data_preprocessing(feature_type, proc_numbers=2)
        print("------------------------------mcdropout_20-----------------------------------")
        print("评价干净的样本")
        mc_20.evaluate(prist_data, prist_y)
        print("------------------------------------------------------------------------")
        print("评价对抗样本")
        mc_20.evaluate(adv_data, adv_y)

        print("------------------------------mcdropout_30-----------------------------------")
        print("评价干净的样本")
        mc_30.evaluate(prist_data, prist_y)
        print("------------------------------------------------------------------------")
        print("评价对抗样本")
        mc_30.evaluate(adv_data, adv_y)

    elif args.data_type == "drebin":
        ood_data, oos_y, input_dim = oos_data_preprocessing(feature_type, proc_numbers=2)

        print("------------------------------mcdropout_20-----------------------------------")
        mc_20.evaluate(ood_data, oos_y)

        print("------------------------------mcdropout_30-----------------------------------")
        mc_30.evaluate(ood_data, oos_y)

    elif args.data_type == "amd":
        ood_data, oos_y, input_dim = oos_data_preprocessing(feature_type, proc_numbers=2, data_type="amd")

        print("------------------------------mcdropout_20-----------------------------------")
        mc_20.evaluate(ood_data, oos_y)

        print("------------------------------mcdropout_30-----------------------------------")
        mc_30.evaluate(ood_data, oos_y)
