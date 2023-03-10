# conduct the group of 'out of distribution' experiments on drebin dataset
import os
import sys
import random
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

from core.feature import feature_type_scope_dict, feature_type_vs_architecture
from core.ensemble import ensemble_method_scope_dict
from core.post_calibration.temperature_scaling import apply_temperature_scaling
from tools import utils
from config import config, logging

logger = logging.getLogger('experiment.drebin_ood')


# procedure of ood experiments
# 1. build dataset
# 2. preprocess data
# 3. conduct prediction
# 4. save results for statistical analysis


def run_experiment(feature_type, ensemble_type, proc_numbers=2):
    """
    run this group of experiments
    :param feature_type: the type of features (e.g., drebin, opcode, etc.), feature type associates to the model architecture
    :param ensemble_type: the ensemble method (e.g., vanilla, deep_ensemble, etc.
    :return: None
    """
    ood_data, ood_y, input_dim = data_preprocessing(feature_type, proc_numbers)

    ensemble_obj = get_ensemble_object(ensemble_type)
    # instantiation
    arch_type = feature_type_vs_architecture.get(feature_type)
    model_saving_dir = config.get('experiments', 'drebin')
    if ensemble_type in ['vanilla', 'mc_dropout', 'bayesian']:
        ensemble_model = ensemble_obj(arch_type, base_model=None, model_directory=model_saving_dir)
    else:
        ensemble_model = ensemble_obj(arch_type, base_model=None, model_directory=model_saving_dir)

    ood_results = ensemble_model.predict(ood_data)
    utils.dump_joblib((ood_results, ood_y), os.path.join(config.get('experiments', 'oos'),
                                                         '{}_{}_drebin_oos.res'.format(feature_type, ensemble_type)))


def run_temperature_scaling(feature_type, ensemble_type, proc_numbers=2):
    ood_data, ood_y, input_dim = data_preprocessing(feature_type, proc_numbers)

    ensemble_obj = get_ensemble_object(ensemble_type)
    # instantiation
    arch_type = feature_type_vs_architecture.get(feature_type)
    model_saving_dir = config.get('experiments', 'drebin')
    ensemble_model = ensemble_obj(arch_type, base_model=None, model_directory=model_saving_dir)
    # temperature scaling

    temp_save_dir = os.path.join(config.get('drebin', 'intermediate_directory'),
                                 "{}_{}_temp.json".format(feature_type, ensemble_type))
    if not os.path.exists(temp_save_dir):
        raise FileNotFoundError

    temperature = utils.load_json(temp_save_dir)['temperature']
    probs = ensemble_model.predict(ood_data, use_prob=True)
    probs_scaling = apply_temperature_scaling(temperature, probs)
    utils.dump_joblib((probs_scaling, ood_y), os.path.join(config.get('experiments', 'oos'),
                                                           '{}_{}_temperature_drebin_oos.res'.format(feature_type,
                                                                                                     ensemble_type)))


def data_preprocessing(feature_type='drebin', proc_numbers=2, data_type="drebin"):
    assert feature_type in feature_type_scope_dict.keys(), 'Expected {}, but {} are supported.'.format(
        feature_type, feature_type_scope_dict.keys())
    benware_dir = config.get('oos', data_type + '_benware_dir')
    malware_dir = config.get('oos', data_type + '_malware_dir')
    android_features_saving_dir = config.get('metadata', 'naive_data_pool')
    intermediate_data_saving_dir = config.get('oos', 'intermediate_directory')
    feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                              intermediate_data_saving_dir,
                                                              update=False,
                                                              proc_number=proc_numbers)

    save_path = os.path.join(intermediate_data_saving_dir, data_type + '_oos_database.' + feature_type)
    if os.path.exists(save_path):
        oos_filenames, oos_y = utils.read_joblib(save_path)
        oos_features = [os.path.join(android_features_saving_dir, filename) for filename in oos_filenames]
    else:
        mal_feature_list = feature_extractor.feature_extraction(malware_dir)
        n_malware = len(mal_feature_list)
        ben_feature_list = feature_extractor.feature_extraction(benware_dir)
        n_benware = len(ben_feature_list)
        oos_features = mal_feature_list + ben_feature_list
        oos_y = np.zeros((n_malware + n_benware,), dtype=np.int32)
        oos_y[:n_malware] = 1

        oos_filenames = [os.path.basename(path) for path in oos_features]
        utils.dump_joblib((oos_filenames, oos_y), save_path)

    # obtain data in a format for ML algorithms
    ood_data, input_dim = feature_extractor.feature2ipt(oos_features)
    return ood_data, oos_y, input_dim


def data_preprocessing_get_array(feature_type='drebin', proc_numbers=2, data_type="drebin"):
    assert feature_type in feature_type_scope_dict.keys(), 'Expected {}, but {} are supported.'.format(
        feature_type, feature_type_scope_dict.keys())
    benware_dir = config.get('oos', data_type + '_benware_dir')
    malware_dir = config.get('oos', data_type + '_malware_dir')
    android_features_saving_dir = config.get('metadata', 'naive_data_pool')
    intermediate_data_saving_dir = config.get('oos', 'intermediate_directory')
    feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                              intermediate_data_saving_dir,
                                                              update=False,
                                                              proc_number=proc_numbers)

    save_path = os.path.join(intermediate_data_saving_dir, data_type + '_oos_database.' + feature_type)
    if os.path.exists(save_path):
        oos_filenames, oos_y = utils.read_joblib(save_path)
        oos_features = [os.path.join(android_features_saving_dir, filename) for filename in oos_filenames]
    else:
        mal_feature_list = feature_extractor.feature_extraction(malware_dir)
        n_malware = len(mal_feature_list)
        ben_feature_list = feature_extractor.feature_extraction(benware_dir)
        n_benware = len(ben_feature_list)
        oos_features = mal_feature_list + ben_feature_list
        oos_y = np.zeros((n_malware + n_benware,), dtype=np.int32)
        oos_y[:n_malware] = 1

        oos_filenames = [os.path.basename(path) for path in oos_features]
        utils.dump_joblib((oos_filenames, oos_y), save_path)

    # obtain data in a format for ML algorithms
    ood_data, input_dim, dataX_np = feature_extractor.feature2ipt_ls(oos_features, oos_y)
    return ood_data, oos_y, input_dim, dataX_np, oos_filenames


def data_preprocessing_get_name(feature_type='drebin', proc_numbers=2, data_type="drebin"):
    assert feature_type in feature_type_scope_dict.keys(), 'Expected {}, but {} are supported.'.format(
        feature_type, feature_type_scope_dict.keys())
    benware_dir = config.get('oos', data_type + '_benware_dir')
    malware_dir = config.get('oos', data_type + '_malware_dir')
    android_features_saving_dir = config.get('metadata', 'naive_data_pool')
    intermediate_data_saving_dir = config.get('oos', 'intermediate_directory')
    feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                              intermediate_data_saving_dir,
                                                              update=False,
                                                              proc_number=proc_numbers)

    save_path = os.path.join(intermediate_data_saving_dir, data_type + '_oos_database.' + feature_type)
    if os.path.exists(save_path):
        oos_filenames, oos_y = utils.read_joblib(save_path)
        oos_features = [os.path.join(android_features_saving_dir, filename) for filename in oos_filenames]
    else:
        mal_feature_list = feature_extractor.feature_extraction(malware_dir)
        n_malware = len(mal_feature_list)
        ben_feature_list = feature_extractor.feature_extraction(benware_dir)
        n_benware = len(ben_feature_list)
        oos_features = mal_feature_list + ben_feature_list
        oos_y = np.zeros((n_malware + n_benware,), dtype=np.int32)
        oos_y[:n_malware] = 1

        oos_filenames = [os.path.basename(path) for path in oos_features]
        utils.dump_joblib((oos_filenames, oos_y), save_path)

    # obtain data in a format for ML algorithms
    ood_data, input_dim = feature_extractor.feature2ipt(oos_features)
    return ood_data, oos_y, input_dim, oos_filenames


def get_ensemble_object(ensemble_type):
    assert ensemble_type in ensemble_method_scope_dict.keys(), '{} expected, but {} are supported'.format(
        ensemble_type,
        ','.join(ensemble_method_scope_dict.keys())
    )
    return ensemble_method_scope_dict[ensemble_type]
