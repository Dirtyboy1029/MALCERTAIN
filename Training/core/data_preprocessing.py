# conduct the group of 'out of distribution' experiments on drebin1 dataset
import os
import pickle
from .feature import feature_type_scope_dict
from .tools import utils
from .config import config

feature_extension = {'drebin': 'drebin',
                     'opcode': 'opcode',
                     'multimodality': 'multimod',
                     'apiseq': 'seq'
                     }


def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


def load_dict_from_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def data_preprocessing(train_data_type='base', test_data_type='2013', feature_type='drebin', is_training_set=True,
                       is_finetune=False):
    intermediate_data_saving_dir = config.get('metadata', 'intermediate_directory')
    if not is_finetune:
        if is_training_set:
            save_path = os.path.join(intermediate_data_saving_dir,
                                     'databases_' + feature_type + '_' + str(train_data_type) + '.conf')
        else:
            save_path = os.path.join(intermediate_data_saving_dir,
                                     'databases_' + feature_type + '_' + str(test_data_type) + '.conf')
    else:
        save_path = os.path.join(intermediate_data_saving_dir,
                                 'databases_finetune_' + feature_type + '_' + str(train_data_type) + '_' + str(
                                     test_data_type) + '.conf')

    print('load filename and label from ' + save_path)
    data_filenames, gt_labels = utils.read_joblib(save_path)

    android_features_saving_dir = config.get('metadata', 'naive_data_pool')

    feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                              intermediate_data_saving_dir,
                                                              update=False,
                                                              proc_number=8)
    oos_features = [os.path.join(android_features_saving_dir, filename) + '.' + feature_extension[feature_type] for
                    filename in
                    data_filenames]

    if is_training_set:
        dataset, input_dim, dataX_np = feature_extractor.feature2ipt(oos_features, gt_labels, data_type=train_data_type)
    else:
        dataset, input_dim, dataX_np = feature_extractor.feature2ipt(oos_features, labels=None,
                                                                     data_type=train_data_type)
    return dataset, gt_labels, input_dim
