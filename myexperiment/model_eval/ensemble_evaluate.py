# -*- coding: utf-8 -*- 
# @Time : 2022/9/19 21:35 
# @Author : DirtyBoy 
# @File : ensemble_evaluate.py
from experiments.oos import data_preprocessing as oos_data_preprocessing
from core.ensemble.deep_ensemble import DeepEnsemble
from experiments.adv import data_preprocessing
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

    den_20_5 = DeepEnsemble(architecture_type=architecture_type,
                            model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/ensemble/n_member5/epoch20")

    den_30_5 = DeepEnsemble(architecture_type=architecture_type,
                            model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/ensemble/n_member5/epoch30")

    den_20_10 = DeepEnsemble(architecture_type=architecture_type,
                             model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/ensemble/n_member10/epoch20")

    den_30_10 = DeepEnsemble(architecture_type=architecture_type,
                             model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/ensemble/n_member10/epoch30")

    if args.data_type == "adv":
        prist_data, adv_data, prist_y, adv_y, input_dim = data_preprocessing(feature_type, proc_numbers=2)

        print("------------------------------epoch_20,n_member5-----------------------------------")
        den_20_5.evaluate(prist_data, prist_y)
        print("------------------------------------------------------------------------")
        den_20_5.evaluate(adv_data, adv_y)

        print("------------------------------epoch_30,n_member5-----------------------------------")
        den_30_5.evaluate(prist_data, prist_y)
        print("------------------------------------------------------------------------")
        den_30_5.evaluate(adv_data, adv_y)

        print("------------------------------epoch_20,n_member10-----------------------------------")
        den_20_10.evaluate(prist_data, prist_y)
        print("------------------------------------------------------------------------")
        den_20_10.evaluate(adv_data, adv_y)

        print("------------------------------epoch_30,n_member10-----------------------------------")
        den_30_10.evaluate(prist_data, prist_y)
        print("------------------------------------------------------------------------")
        den_30_10.evaluate(adv_data, adv_y)

    elif args.data_type == "drebin":
        ood_data, oos_y, input_dim = oos_data_preprocessing(feature_type, proc_numbers=2)

        print("------------------------------epoch_20,n_member5-----------------------------------")
        den_20_5.evaluate(ood_data, oos_y)

        print("------------------------------epoch_30,n_member5-----------------------------------")
        den_30_5.evaluate(ood_data, oos_y)

        print("------------------------------epoch_20,n_member10-----------------------------------")
        den_20_10.evaluate(ood_data, oos_y)

        print("------------------------------epoch_30,n_member10-----------------------------------")
        den_30_10.evaluate(ood_data, oos_y)

    elif args.data_type == "amd":
        ood_data, oos_y, input_dim = oos_data_preprocessing(feature_type, proc_numbers=2, data_type="amd")

        print("------------------------------epoch_20,n_member5-----------------------------------")
        den_20_5.evaluate(ood_data, oos_y)

        print("------------------------------epoch_30,n_member5-----------------------------------")
        den_30_5.evaluate(ood_data, oos_y)

        print("------------------------------epoch_20,n_member10-----------------------------------")
        den_20_10.evaluate(ood_data, oos_y)

        print("------------------------------epoch_30,n_member10-----------------------------------")
        den_30_10.evaluate(ood_data, oos_y)
