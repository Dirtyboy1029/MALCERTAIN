# -*- coding: utf-8 -*- 
# @Time : 2024/9/23 13:02 
# @Author : DirtyBoy 
# @File : predict.py
from core.data_preprocessing import data_preprocessing
from core.ensemble.vanilla import Vanilla
from core.ensemble.bayesian_ensemble import BayesianEnsemble
from core.ensemble.deep_ensemble import DeepEnsemble
from core.ensemble.mc_dropout import MCDropout
from core.ensemble.deep_ensemble import WeightedDeepEnsemble
import argparse, os
import numpy as np

# Model_path = "../Model/"
Model_path = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/MalCertain/Models/'

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
    parser.add_argument('-test_data_type', '-tdt', type=str, default="correct")
    parser.add_argument('-data_type', '-dt', type=str, default="base")
    parser.add_argument('-feature_type', '-ft', type=str, default="drebin")
    parser.add_argument('-experiment_type', '-et', type=str, default="predict")
    args = parser.parse_args()
    experiment_type = args.experiment_type
    feature_type = args.feature_type
    data_type = args.data_type
    test_data_type = args.test_data_type
    architecture_type = Architecture_Type[feature_type]

    test_dataset, gt_labels, input_dim = data_preprocessing(feature_type=feature_type, train_data_type=data_type,
                                                            test_data_type=test_data_type,
                                                            is_training_set=False, is_finetune=False)
    output_path = 'output/' + feature_type
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if experiment_type == 'uncertainty':
        for model_type in ['vanilla', 'epoch_ensemble', 'bayesian', 'mc_dropout', 'ensemble', 'wensemble']:
            if model_type == 'epoch_ensemble':
                models = [Vanilla(architecture_type=architecture_type,
                                  model_directory=Model_path + feature_type + '/' + data_type + '/epoch_ensemble/' + str(
                                      i + 1)) for i in range(10)]
                prob_set = []
                for model in models:
                    prob = model.predict(test_dataset, use_prob=True)
                    prob_set.append(prob)
                prob_set = np.array(prob_set)
                prob_set = prob_set.transpose(1, 0, 2)
                np.save(os.path.join(output_path, data_type + '_' + model_type + '_' + test_data_type), prob_set)
            else:
                model_ensemble = Model_Type[model_type]
                model = model_ensemble(architecture_type=architecture_type,
                                       model_directory=Model_path + feature_type + '/' + data_type + '/')
                prob = model.predict(test_dataset)
                if model_type == 'wensemble':
                    np.save(os.path.join(output_path, data_type + '_' + model_type + '_' + test_data_type), prob[0])
                    np.save(os.path.join(output_path, data_type + '_' + model_type + '_' + test_data_type + '_weights'),
                            prob[1])
                else:
                    np.save(os.path.join(output_path, data_type + '_' + model_type + '_' + test_data_type), prob)
    elif experiment_type == 'predict':
        model_type = 'vanilla'
        model_ensemble = Model_Type[model_type]
        model = model_ensemble(architecture_type=architecture_type,
                               model_directory=Model_path + feature_type + '/' + data_type + '/')
        prob = model.predict(test_dataset)
        np.save(os.path.join(output_path, data_type + '_' + model_type + '_' + test_data_type), prob)
    else:
        print('error!!')

    ### CUDA_VISIBLE_DEVICES
