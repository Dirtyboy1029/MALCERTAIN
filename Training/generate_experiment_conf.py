# -*- coding: utf-8 -*- 
# @Time : 2024/9/20 8:36 
# @Author : DirtyBoy 
# @File : generate_experiment_conf.py
import pandas as pd
import os, random
import numpy as np
from core.config import config


def dump_joblib(data, path):
    try:
        import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def remove_inter_file_not_exist(df, feature_type):
    directory = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/naive_pool'

    def check_file_exists(hash_value):
        file_path = os.path.join(directory, f"{hash_value}." + feature_extension[feature_type])
        return os.path.exists(file_path)

    df['file_exists'] = df['sha256'].apply(check_file_exists)
    df1 = df[df['file_exists']]
    df2 = df[~df['file_exists']]
    df1 = df1.drop(columns=['file_exists'])
    return df1, df2


def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


feature_extension = {'drebin': 'drebin',
                     'opcode': 'opcode',
                     'multimodality': 'multimod',
                     'apiseq': 'seq'
                     }

time_lines = {'20120601': 'base',
              '20120801': 'correct',
              '20130101': '2013',
              '20160101': '2016',
              '20230101': 'latest'}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_type', '-dt', type=str, default='baseplus')
    parser.add_argument('-feature_type', '-ft', type=str, default='opcode')
    args = parser.parse_args()
    data_type = args.data_type
    feature_type = args.feature_type

    if data_type == 'baseplus':
        base_data_filenames, base_labels = read_joblib(f'config/databases_{feature_type}_base.conf')
        correct_data_filenames, correct_labels = read_joblib(f'config/databases_{feature_type}_correct.conf')
        data_filenames = base_data_filenames + correct_data_filenames
        labels = np.array(list(base_labels) + list(correct_labels))
        dump_joblib((data_filenames, labels),
                    'config/databases_' + feature_type+'tmp' + '_' + data_type + '.conf')

    else:
        def main_(data_type, feature_type):
            benign_base = pd.read_csv('../Datasets/database/benign/benign_' + data_type + '.csv')
            malware_base = pd.read_csv('../Datasets/database/malware/malware_' + data_type + '.csv')
            benign_base, benign_base1 = remove_inter_file_not_exist(benign_base, feature_type)
            # save_to_txt(benign_base1['sha256'].tolist(), '1.txt')
            malware_base, _ = remove_inter_file_not_exist(malware_base, feature_type)

            benign_base_hash = benign_base['sha256'].tolist()
            print(len(benign_base_hash))
            malware_base_hash = malware_base['sha256'].tolist()
            print(len(malware_base_hash))
            if len(benign_base_hash) > len(malware_base_hash):
                benign_base_hash = random.sample(benign_base_hash, k=len(malware_base_hash))
            else:
                malware_base_hash = random.sample(malware_base_hash, k=len(benign_base_hash))
            print(len(malware_base_hash))
            data_filenames = malware_base_hash + benign_base_hash
            labels = np.concatenate((np.ones(len(malware_base_hash)), np.zeros(len(benign_base_hash))), axis=0)
            print(len(data_filenames))
            if not os.path.isfile('config/databases_' + feature_type + '_' + time_lines[data_type] + '.conf'):
                dump_joblib((data_filenames, labels),
                            'config/databases_' + feature_type + '_' + time_lines[data_type] + '.conf')
                print('save file to config/databases_' + feature_type + '_' + time_lines[data_type] + '.conf')
            else:
                print('config/databases_' + feature_type + '_' + time_lines[data_type] + '.conf is exist!!!')


        for data_type in time_lines.keys():
            main_(data_type, feature_type)
