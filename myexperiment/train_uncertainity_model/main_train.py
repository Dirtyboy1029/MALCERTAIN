# -*- coding: utf-8 -*- 
# @Time : 2022/10/7 12:03 
# @Author : DirtyBoy 
# @File : main_train.py
from experiments.drebin_dataset import data_preprocessing
from core.ensemble.bayesian_ensemble import BayesianEnsemble
import argparse
from core.ensemble.mc_dropout import MCDropout
from core.ensemble.vanilla import Vanilla
from core.ensemble.deep_ensemble import DeepEnsemble
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

    train_dataset, test_dataset, test_y, input_dim = data_preprocessing(feature_type=feature_type,
                                                                        malware_dir=mal_folder,
                                                                        benware_dir=ben_folder, model_type=model_type)

    bay_ens20 = BayesianEnsemble(architecture_type=architecture_type,
                                 model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/bayesian/epoch20")

    bay_ens20.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=20)

    MSG20 = "20--" + bay_ens20.evaluate(test_dataset,test_y)

    bay_ens30 = BayesianEnsemble(architecture_type=architecture_type,
                                 model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/bayesian/epoch30")

    bay_ens30.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=30)

    MSG30 = "30--" + bay_ens30.evaluate(test_dataset,test_y)

    MSG = MSG20 + "\n" + MSG30

    with open('/home/lhd/uncertainity_malware/myexperiment/output/eval_model_log/' + model_type + '_bayesian.txt',
              'w') as f:
        f.write(MSG)

    d_ensemble1020 = DeepEnsemble(architecture_type=architecture_type, n_members=10,
                                  model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/ensemble/n_member10/epoch20")

    d_ensemble1020.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=20)

    msg1020 = "1020--" + d_ensemble1020.evaluate(test_dataset,test_y)

    d_ensemble1030 = DeepEnsemble(architecture_type=architecture_type, n_members=10,
                                  model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/ensemble/n_member10/epoch30")

    d_ensemble1030.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=30)

    msg1030 = "1030--" + d_ensemble1030.evaluate(test_dataset,test_y)

    d_ensemble520 = DeepEnsemble(architecture_type=architecture_type, n_members=5,
                                 model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/ensemble/n_member5/epoch20")

    d_ensemble520.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=20)

    msg520 = "520--" + d_ensemble520.evaluate(test_dataset,test_y)

    d_ensemble530 = DeepEnsemble(architecture_type=architecture_type, n_members=5,
                                 model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/ensemble/n_member5/epoch30")

    d_ensemble530.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=30)

    msg530 = "530--" + d_ensemble530.evaluate(test_dataset,test_y)

    MSG = msg520 + "\n" + msg530 + "\n" + msg1030 + "\n" + msg1020

    with open(
            '/home/lhd/uncertainity_malware/myexperiment/output/eval_model_log/' + model_dir + model_type + '_ensemble.txt',
            'w') as f:
        f.write(MSG)

    mc_dp20 = MCDropout(architecture_type=architecture_type,
                        model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/mc_dropout/epoch20")

    mc_dp20.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=20)

    MSG20 = "20--" + mc_dp20.evaluate(test_dataset,test_y)

    mc_dp30 = MCDropout(architecture_type=architecture_type,
                        model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/mc_dropout/epoch30")

    mc_dp30.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=30)

    MSG30 = "30--" + mc_dp30.evaluate(test_dataset,test_y)

    MSG = MSG20 + "\n" + MSG30

    with open(
            '/home/lhd/uncertainity_malware/myexperiment/output/eval_model_log/' + model_dir + model_type + '_mcdropout.txt',
            'w') as f:
        f.write(MSG)

    MSG = ""

    for epoch in range(10, 60, 10):
        vanilla = Vanilla(architecture_type=architecture_type,
                          model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/vanilla/epoch" + str(
                              epoch))

        vanilla.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=epoch)

        MSG_ = str(epoch) + "--" + vanilla.evaluate(test_dataset,test_y)

        MSG = MSG + MSG_ + "\n"

    with open(
            '/home/lhd/uncertainity_malware/myexperiment/output/eval_model_log/' + model_dir + model_type + '_vanilla.txt',
            'w') as f:
        f.write(MSG)

    w_ensemble1020 = WeightedDeepEnsemble(architecture_type=architecture_type, n_members=10,
                                          model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/wensemble/n_member10/epoch20")

    w_ensemble1020.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=20)

    msg1020 = "1020--" + w_ensemble1020.evaluate(test_dataset,test_y)

    w_ensemble520 = WeightedDeepEnsemble(architecture_type=architecture_type, n_members=5,
                                         model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/wensemble/n_member5/epoch20")

    w_ensemble520.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=20)

    msg520 = "520--" + w_ensemble520.evaluate(test_dataset,test_y)

    w_ensemble530 = WeightedDeepEnsemble(architecture_type=architecture_type, n_members=5,
                                         model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/wensemble/n_member5/epoch30")

    w_ensemble530.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=30)

    msg530 = "530--" + w_ensemble530.evaluate(test_dataset,test_y)

    w_ensemble1030 = WeightedDeepEnsemble(architecture_type=architecture_type, n_members=10,
                                          model_directory="/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/wensemble/n_member10/epoch30")

    w_ensemble1030.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=30)

    msg1030 = "1030--" + w_ensemble1030.evaluate(test_dataset,test_y)

    MSG = msg520 + "\n" + msg530 + "\n" + msg1030 + "\n" + msg1020

    with open(
            '/home/lhd/uncertainity_malware/myexperiment/output/eval_model_log/' + model_dir + model_type + '_wensemble.txt',
            'w') as f:
        f.write(MSG)
