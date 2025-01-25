# -*- coding: utf-8 -*- 
# @Time : 2024/9/25 9:32 
# @Author : DirtyBoy 
# @File : evaluate_malcertain.py
import numpy as np
import os, argparse
from myexperiments.utils import evaluate, read_joblib, load_correction_model, correct_model_training_data_processing, \
    evaluate_correction

parser = argparse.ArgumentParser()
parser.add_argument('-test_data_type', '-tdt', type=str, default="2014")
parser.add_argument('-data_type', '-dt', type=str, default="finetune_2013")
parser.add_argument('-feature_type', '-ft', type=str, default="drebin")
args = parser.parse_args()
feature_type = args.feature_type
data_type = args.data_type
test_data_type = args.test_data_type

vanilla_prob = np.squeeze(
    np.mean(np.load(
        '../Training/output/' + feature_type + '/' + data_type + '_vanilla_' + test_data_type + '.npy'),
        axis=1))

save_path = os.path.join('../Training/config',
                         'databases_' + feature_type + '_' + test_data_type + '.conf')

print('load filename and label from ' + save_path)
data_filenames, gt_labels = read_joblib(save_path)
print(data_type, test_data_type)
evaluate(vanilla_prob, gt_labels)

correction_model = load_correction_model('correction_model/' + feature_type + '_base_correct_correct.pkl')

feature, true_index, false_index = correct_model_training_data_processing(feature_type, data_type,
                                                                          test_data_type)

is_flase = correction_model.predict(feature)
#np.save('malcertain_mask/' + feature_type + '_' + data_type + '_mask_' + test_data_type + '.npy', is_flase)
evaluate_correction(vanilla_prob, gt_labels, is_flase, only_fn=True)
# evaluate_correction(vanilla_prob, gt_labels, is_flase, only_fn=False)
