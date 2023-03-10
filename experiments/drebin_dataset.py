# conduct the first group experiments on drebin dataset
import os

import numpy as np
from sklearn.model_selection import train_test_split

from core.feature import feature_type_scope_dict, feature_type_vs_architecture
from core.ensemble import ensemble_method_scope_dict
from tools import utils
from config import config, logging

logger = logging.getLogger('experiment.drebin')


# procedure of drebin experiments
# 1. build dataset
# 2. preprocess data
# 3. learn models
# 4. save results for statistical analysis

def run_experiment(feature_type, ensemble_type, random_seed=0, n_members=1, ratio=3.0, proc_numbers=2):
    """
    run this group of experiments
    :param feature_type: the type of feature (e.g., drebin, opcode, etc.), feature type associates to the model architecture
    :param ensemble_type: the ensemble method (e.g., vanilla, deep_ensemble, etc.
    :param random_seed: an integer
    :param n_members: the number of base models enclosed by an ensemble
    :param ratio: the ratio of benign files to malware files
    :param proc_numbers: the number of threads
    :return: None
    """
    mal_folder = config.get('drebin', 'malware_dir')
    ben_folder = config.get('drebin', 'benware_dir')
    logger.info('testing:{},{}'.format(feature_type, ensemble_type))
    logger.info('The seed is :{}'.format(random_seed))

    train_dataset, validation_dataset, test_data, test_y, input_dim = \
        data_preprocessing(feature_type, mal_folder, ben_folder, ratio, proc_numbers, random_seed)

    ensemble_obj = get_ensemble_object(ensemble_type)
    # instantiation
    arch_type = feature_type_vs_architecture.get(feature_type)
    saving_dir = config.get('experiments', 'drebin')
    if ensemble_type in ['vanilla', 'mc_dropout', 'bayesian']:
        ensemble_model = ensemble_obj(arch_type, base_model=None, n_members=1, model_directory=saving_dir)
    else:
        ensemble_model = ensemble_obj(arch_type, base_model=None, n_members=n_members, model_directory=saving_dir)

    ensemble_model.fit(train_dataset, validation_dataset, input_dim=input_dim)

    test_results = ensemble_model.predict(test_data)
    utils.dump_joblib((test_results, test_y),
                      os.path.join(saving_dir, '{}_{}_test.res'.format(feature_type, ensemble_type)))
    ensemble_model.evaluate(test_data, test_y)
    return


def run_temperature_scaling(feature_type, ensemble_type):
    """
    Run temperature scaling
    :param feature_type: the type of feature (e.g., drebin, opcode, etc.), feature type associates to the model architecture
    :param ensemble_type: the ensemble method (e.g., vanilla, deep_ensemble, etc.
    """
    from core.post_calibration.temperature_scaling import find_scaling_temperature, \
        apply_temperature_scaling, inverse_sigmoid
    logger.info('run temperature scaling:{},{}'.format(feature_type, ensemble_type))

    # load dataset
    def data_load(feature_type='drebin'):
        assert feature_type in feature_type_scope_dict.keys(), 'Expected {}, but {} are supported.'.format(
            feature_type, list(feature_type_scope_dict.keys()))

        android_features_saving_dir = config.get('metadata', 'naive_data_pool')
        intermediate_data_saving_dir = config.get('drebin', 'intermediate_directory')
        feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                                  intermediate_data_saving_dir,
                                                                  update=False,
                                                                  proc_number=1)

        save_path = os.path.join(intermediate_data_saving_dir, 'drebin_database.' + feature_type)
        if os.path.exists(save_path):
            _1, test_filenames, validation_filenames, \
            _2, test_y, validation_y = utils.read_joblib(save_path)
            validation_features = [os.path.join(android_features_saving_dir, filename) for filename in
                                   validation_filenames]
            test_features = [os.path.join(android_features_saving_dir, filename) for filename in
                             test_filenames]
        else:
            raise ValueError

        test_data, _ = feature_extractor.feature2ipt(test_features)
        validation_data, _ = feature_extractor.feature2ipt(validation_features)

        return validation_data, test_data, validation_y, test_y

    mal_folder = config.get('drebin', 'malware_dir')
    ben_folder = config.get('drebin', 'benware_dir')
    val_data, test_data, val_label, test_y = \
        data_load(feature_type, mal_folder, ben_folder)

    # load model
    ensemble_obj = get_ensemble_object(ensemble_type)
    arch_type = feature_type_vs_architecture.get(feature_type)
    model_saving_dir = config.get('experiments', 'drebin')
    if ensemble_type in ['vanilla', 'mc_dropout', 'bayesian']:
        ensemble_model = ensemble_obj(arch_type, base_model=None, model_directory=model_saving_dir)
    else:
        ensemble_model = ensemble_obj(arch_type, base_model=None, model_directory=model_saving_dir)

    temp_save_dir = os.path.join(config.get('drebin', 'intermediate_directory'),
                                 "{}_{}_temp.json".format(feature_type, ensemble_type))
    if not os.path.exists(temp_save_dir):
        prob = np.squeeze(ensemble_model.predict(val_data, use_prob=True))
        logits = inverse_sigmoid(prob)
        temperature = find_scaling_temperature(val_label, logits)
        utils.dump_json({'temperature': temperature}, temp_save_dir)
    temperature = utils.load_json(temp_save_dir)['temperature']

    prob_test = ensemble_model.predict(test_data, use_prob=True)
    prob_t = apply_temperature_scaling(temperature, prob_test)
    utils.dump_joblib((prob_t, test_y),
                      os.path.join(model_saving_dir, '{}_{}_temperature_test.res'.format(feature_type, ensemble_type)))


def data_preprocessing(feature_type='drebin', malware_dir=None, benware_dir=None, ratio=3.0, proc_numbers=2,
                       random_seed=0, model_type="big"):
    assert feature_type in feature_type_scope_dict.keys(), 'Expected {}, but {} are supported.'.format(
        feature_type, list(feature_type_scope_dict.keys()))

    android_features_saving_dir = config.get('metadata', 'naive_data_pool')
    # "/home/lhd/apk/native"
    # config.get('metadata', 'naive_data_pool')
    intermediate_data_saving_dir = config.get('drebin', 'intermediate_directory')
    feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                              intermediate_data_saving_dir,
                                                              update=False,
                                                              proc_number=proc_numbers)
    save_path = os.path.join(intermediate_data_saving_dir, model_type + '_drebin_database.' + feature_type)

    if os.path.exists(save_path):

        train_filenames, test_filenames, \
        train_y, test_y = utils.read_joblib(save_path)

        train_features = [os.path.join(config.get('metadata', 'naive_data_pool'), filename) for filename in
                          train_filenames]
        # validation_features = [os.path.join(config.get('metadata', 'naive_data_pool'), filename) for filename in
        # validation_filenames]
        test_features = [os.path.join(config.get('metadata', 'naive_data_pool'), filename) for filename in
                         test_filenames]

    else:
        def train_test_val_split(data):
            train, test = train_test_split(data, test_size=0.2, random_state=random_seed)
            # train, val = train_test_split(train, test_size=0.25, random_state=random_seed)
            return train, test

        def merge_mal_ben(mal, ben):
            mal_feature_list = feature_extractor.feature_extraction(mal)
            n_malware = len(mal_feature_list)
            ben_feature_list = feature_extractor.feature_extraction(ben)
            n_benware = len(ben_feature_list)
            feature_list = mal_feature_list + ben_feature_list
            gt_labels = np.zeros((n_malware + n_benware,), dtype=np.int32)
            gt_labels[:n_malware] = 1
            import random
            random.seed(0)
            random.shuffle(feature_list)
            random.seed(0)
            random.shuffle(gt_labels)
            return feature_list, gt_labels

        malware_path_list = utils.retrive_files_set(malware_dir, "", ".apk|")
        mal_train, mal_test = train_test_val_split(malware_path_list)
        benware_path_list = utils.retrive_files_set(benware_dir, "", ".apk|")
        ben_train, ben_test = train_test_val_split(benware_path_list)

        # undersampling the benign files
        if ratio >= 1.:
            ben_train = undersampling(ben_train, len(mal_train), ratio)
        logger.info('Training set, the number of benign files vs. malware files: {} vs. {}'.format(len(ben_train),
                                                                                                   len(mal_train)))
        train_features, train_y = merge_mal_ben(mal_train, ben_train)
        # validation_features, validation_y = merge_mal_ben(mal_val, ben_val)
        test_features, test_y = merge_mal_ben(mal_test, ben_test)

        train_filenames = [os.path.basename(path) for path in train_features]
        # validation_filenames = [os.path.basename(path) for path in validation_features]
        test_filenames = [os.path.basename(path) for path in test_features]
        utils.dump_joblib(
            (train_filenames, test_filenames, train_y, test_y),
            save_path)
    # obtain data in a format for ML algorithms
    feature_extractor.feature_preprocess(train_features, train_y)  # produce datasets products

    train_dataset, input_dim, train_data_np = feature_extractor.feature2ipt_ls(train_features, train_y,
                                                                               is_training_set=True)
    test_dataset, input_dim, test_data_np = feature_extractor.feature2ipt_ls(test_features)
    #test_dataset, input_dim = feature_extractor.feature2ipt(test_features)

    return train_dataset, test_dataset, test_y, input_dim, train_data_np, test_data_np,train_y


def undersampling(ben_train, num_of_mal, ratio):
    number_of_choice = int(num_of_mal * ratio) if int(num_of_mal * ratio) <= len(ben_train) else len(ben_train)
    import random
    random.seed(0)
    random.shuffle(ben_train)
    return ben_train[:number_of_choice]


def get_ensemble_object(ensemble_type):
    assert ensemble_type in ensemble_method_scope_dict.keys(), '{} expected, but {} are supported'.format(
        ensemble_type,
        ','.join(ensemble_method_scope_dict.keys())
    )
    return ensemble_method_scope_dict[ensemble_type]
