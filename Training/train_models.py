# -*- coding: utf-8 -*- 
# @Time : 2024/9/20 17:13 
# @Author : DirtyBoy 
# @File : train_models.py
from core.data_preprocessing import data_preprocessing
from core.ensemble.vanilla import Vanilla
from core.ensemble.bayesian_ensemble import BayesianEnsemble
from core.ensemble.deep_ensemble import DeepEnsemble
from core.ensemble.mc_dropout import MCDropout
from core.ensemble.deep_ensemble import WeightedDeepEnsemble
import argparse

Model_Type = {'vanilla': Vanilla,
              'bayesian': BayesianEnsemble,
              'mc_dropout': MCDropout,
              'ensemble': DeepEnsemble,
              'wensemble': WeightedDeepEnsemble
              }

Architecture_Type = {'drebin': 'dnn',
                     'opcode': 'text_cnn',
                     'apiseq': 'droidectc',
                     'multimodality': 'multimodalitynn'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_type', '-tt', type=str, default="pretrain")
    parser.add_argument('-data_type', '-dt', type=str, default="base")
    parser.add_argument('-finetune_type', '-ftt', type=str, default="2013")
    parser.add_argument('-model_type', '-mt', type=str, default="vanilla")
    parser.add_argument('-feature_type', '-ft', type=str, default="drebin")
    args = parser.parse_args()
    model_type = args.model_type
    feature_type = args.feature_type
    data_type = args.data_type
    finetune_type = args.finetune_type
    train_type = args.train_type
    architecture_type = Architecture_Type[feature_type]
    try:
        model_ensemble = Model_Type[model_type]
    except Exception as e:
        print(e)
        model_ensemble = Vanilla

    if model_type == 'epoch_ensemble':
        model = model_ensemble(architecture_type=architecture_type,
                               model_directory="../Model/" + feature_type + '/' + data_type + '/epoch_ensemble/')
    else:
        model = model_ensemble(architecture_type=architecture_type,
                               model_directory="../Model/" + feature_type + '/' + data_type + '/')
    if train_type == 'pretrain':
        train_dataset, gt_labels, input_dim = data_preprocessing(feature_type=feature_type, train_data_type=data_type)
        if model_type == 'epoch_ensemble':
            for i in range(10):
                if i == 0:
                    model.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=5)
                else:
                    model.finetune(train_set=train_dataset, input_dim=input_dim, EPOCH=5)
                model.save_ensemble_weights(
                    save_path="../Model/" + feature_type + '/' + data_type + '/epoch_ensemble/' + str(i + 1))
        else:
            model.fit(train_set=train_dataset, input_dim=input_dim, EPOCH=30)
    elif train_type == 'finetune':
        train_dataset, gt_labels, input_dim = data_preprocessing(feature_type=feature_type,
                                                                 train_data_type=data_type,
                                                                 test_data_type=finetune_type, is_finetune=True)
        model.finetune(train_set=train_dataset, input_dim=input_dim, EPOCH=1)
        model.save_dir = "../Model/" + feature_type + '/finetune_' + finetune_type + '/' + model.name
        model.save_ensemble_weights()

    ### CUDA_VISIBLE_DEVICES
