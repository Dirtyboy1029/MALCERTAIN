# -*- coding: utf-8 -*- 
# @Time : 2024/9/20 20:01 
# @Author : DirtyBoy 
# @File : generate_finetune_conf.py
import pandas as pd
import os, random
import numpy as np
from core.config import config


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def dump_joblib(data, path):
    try:
        import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


def remove_inter_file_not_exist(df, feature_type):
    directory = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/naive_pool'

    def check_file_exists(hash_value):
        file_path = os.path.join(directory, f"{hash_value}." + feature_type)
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


time_lines = {'20120601': 'base',
              '20120801': 'correct',
              '20130801': '2013',
              '20140801': '2014',
              '20160801': '2016',
              '20220801': '2022'}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-feature_type', '-ft', type=str, default='opcode')
    args = parser.parse_args()
    feature_type = args.feature_type
    benign_base = pd.read_csv('../Datasets/database/benign/benign_20130801.csv')

    benign_base, benign_base1 = remove_inter_file_not_exist(benign_base, feature_type)

    benign_base_hash = benign_base['sha256'].tolist()
    print(len(benign_base_hash))

    vanilla_prob = np.squeeze(
        np.mean(np.load(
            '../Training/output/' + feature_type + '/base_vanilla_2013.npy'),
            axis=1))

    save_path = os.path.join('../Training/config',
                             'databases_' + feature_type + '_2013.conf')

    print('load filename and label from ' + save_path)
    data_filenames, gt_labels = read_joblib(save_path)
    vanilla_pred = (vanilla_prob >= 0.5).astype(np.int32)

    false_index = list(
        np.where(np.load('../myexperiments/malcertain_mask/' + feature_type + '_base_mask_2013.npy') == 1)[0])
    print(len(false_index))
    false_index = [item for item in false_index if vanilla_pred[item] == 0]
    print(len(false_index))
    malware_base_hash = [data_filenames[item] for item in false_index]
    benign_base_hash = random.sample(benign_base_hash, len(malware_base_hash))
    data_filenames = malware_base_hash + benign_base_hash
    labels = np.concatenate((np.ones(len(malware_base_hash)), np.zeros(len(benign_base_hash))), axis=0)
    print(len(data_filenames))
    dump_joblib((data_filenames, labels),
                'config/databases_finetune_' + feature_type + '_' + time_lines['20130801'] + '.conf')
