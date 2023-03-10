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

logger = logging.getLogger('experiment.drebin_adv')


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
    prist_data, adv_data, prist_y, adv_y, input_dim = data_preprocessing(feature_type, proc_numbers)

    ensemble_obj = get_ensemble_object(ensemble_type)
    # instantiation
    arch_type = feature_type_vs_architecture.get(feature_type)
    model_saving_dir = config.get('experiments', 'drebin')
    if ensemble_type in ['vanilla', 'mc_dropout', 'bayesian']:
        ensemble_model = ensemble_obj(arch_type, base_model=None, model_directory=model_saving_dir)
    else:
        ensemble_model = ensemble_obj(arch_type, base_model=None, model_directory=model_saving_dir)

    prist_results = ensemble_model.predict(prist_data)
    adv_results = ensemble_model.predict(adv_data)
    utils.dump_joblib((prist_results, adv_results, prist_y, adv_y),
                      os.path.join(config.get('experiments', 'adv_eval'),
                                   '{}_{}_drebin_adv.res'.format(feature_type, ensemble_type)))
    # ensemble_model.evaluate(prist_data, prist_y)
    # ensemble_model.evaluate(adv_data, adv_y)


def run_temperature_scaling(feature_type, ensemble_type, proc_numbers=2):
    prist_data, adv_data, prist_y, adv_y, input_dim = data_preprocessing(feature_type, proc_numbers)

    ensemble_obj = get_ensemble_object(ensemble_type)
    # instantiation 
    arch_type = feature_type_vs_architecture.get(feature_type)
    model_saving_dir = config.get('experiments', 'drebin')
    ensemble_model = ensemble_obj(arch_type, base_model=None, model_directory=model_saving_dir)

    temp_save_dir = os.path.join(config.get('drebin', 'intermediate_directory'),
                                 "{}_{}_temp.json".format(feature_type, ensemble_type))
    if not os.path.exists(temp_save_dir):
        raise FileNotFoundError

    temperature = utils.load_json(temp_save_dir)['temperature']

    prist_probs = ensemble_model.predict(prist_data, use_prob=True)
    prist_prob_t = apply_temperature_scaling(temperature, prist_probs)
    adv_probs = ensemble_model.predict(adv_data, use_prob=True)
    adv_prob_t = apply_temperature_scaling(temperature, adv_probs)

    utils.dump_joblib((prist_prob_t, adv_prob_t, prist_y, adv_y), os.path.join(config.get('experiments', 'adv_eval'),
                                                                               '{}_{}_temperature_drebin_adv.res'.format(
                                                                                   feature_type, ensemble_type)))


def data_preprocessing(feature_type='drebin', proc_numbers=2, pristine_apk_dir="pristine_apk_dir",
                       perturbed_apk_dir='perturbed_apk_dir'):
    assert feature_type in feature_type_scope_dict.keys(), 'Expected {}, but {} are supported.'.format(
        feature_type, feature_type_scope_dict.keys())
    malware_pst_dir = config.get('adv', pristine_apk_dir)
    # malware_adv_dir = os.path.join('adv_eval', 'perturbed_apk_dir')
    malware_adv_dir = config.get('adv', perturbed_apk_dir)
    android_features_saving_dir = config.get('metadata', 'naive_data_pool')
    intermediate_data_saving_dir = config.get('adv', 'intermediate_directory')
    feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                              intermediate_data_saving_dir,
                                                              update=False,
                                                              proc_number=proc_numbers)

    save_path = os.path.join(intermediate_data_saving_dir, 'adv_database.' + feature_type)
    if os.path.exists(save_path):
        prist_filenames, adv_filenames, prist_y, adv_y = utils.read_joblib(save_path)
        prist_features = [os.path.join(android_features_saving_dir, filename) for filename in prist_filenames]
        adv_features = [os.path.join(android_features_saving_dir, filename) for filename in adv_filenames]
    else:
        prist_features = feature_extractor.feature_extraction(malware_pst_dir)
        prist_y = np.ones((len(prist_features),), dtype=np.int32)

        adv_features = feature_extractor.feature_extraction(malware_adv_dir)
        adv_y = np.ones((len(adv_features),), dtype=np.int32)

        prist_filenames = [os.path.basename(path) for path in prist_features]
        adv_filenames = [os.path.basename(path) for path in adv_features]
        utils.dump_joblib((prist_filenames, adv_filenames, prist_y, adv_y), save_path)

    # obtain data in a format for ML algorithms
    prist_data, input_dim = feature_extractor.feature2ipt(prist_features)
    adv_data, input_dim = feature_extractor.feature2ipt(adv_features)
    return prist_data, adv_data, prist_y, adv_y, input_dim  # ,prist_data_np,adv_data_np


def get_ensemble_object(ensemble_type):
    assert ensemble_type in ensemble_method_scope_dict.keys(), '{} expected, but {} are supported'.format(
        ensemble_type,
        ','.join(ensemble_method_scope_dict.keys())
    )
    return ensemble_method_scope_dict[ensemble_type]


def _main():
    # build_data()
    print(get_ensemble_object('vanilla'))


def data_preprocessing_get_name(feature_type='drebin', proc_numbers=2,pristine_apk_dir="pristine_apk_dir",
                       perturbed_apk_dir='perturbed_apk_dir'):
    assert feature_type in feature_type_scope_dict.keys(), 'Expected {}, but {} are supported.'.format(
        feature_type, feature_type_scope_dict.keys())
    malware_pst_dir = config.get('adv', pristine_apk_dir)
    # malware_adv_dir = os.path.join('adv_eval', 'perturbed_apk_dir')
    malware_adv_dir = config.get('adv', perturbed_apk_dir)
    android_features_saving_dir = config.get('metadata', 'naive_data_pool')
    intermediate_data_saving_dir = config.get('adv', 'intermediate_directory')
    feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                              intermediate_data_saving_dir,
                                                              update=False,
                                                              proc_number=proc_numbers)

    save_path = os.path.join(intermediate_data_saving_dir, 'adv_database.' + feature_type)
    if os.path.exists(save_path):
        prist_filenames, adv_filenames, prist_y, adv_y = utils.read_joblib(save_path)
        prist_features = [os.path.join(android_features_saving_dir, filename) for filename in prist_filenames]
        adv_features = [os.path.join(android_features_saving_dir, filename) for filename in adv_filenames]
    else:
        prist_features = feature_extractor.feature_extraction(malware_pst_dir)
        prist_y = np.ones((len(prist_features),), dtype=np.int32)

        adv_features = feature_extractor.feature_extraction(malware_adv_dir)
        adv_y = np.ones((len(adv_features),), dtype=np.int32)

        prist_filenames = [os.path.basename(path) for path in prist_features]
        adv_filenames = [os.path.basename(path) for path in adv_features]
        utils.dump_joblib((prist_filenames, adv_filenames, prist_y, adv_y), save_path)

    # obtain data in a format for ML algorithms
    prist_data, input_dim = feature_extractor.feature2ipt(prist_features)
    adv_data, input_dim = feature_extractor.feature2ipt(adv_features)
    return prist_data, adv_data, prist_y, adv_y, input_dim, prist_filenames, adv_filenames


if __name__ == '__main__':
    sys.exit(_main())
