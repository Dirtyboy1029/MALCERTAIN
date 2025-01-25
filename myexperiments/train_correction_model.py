# -*- coding: utf-8 -*- 
# @Time : 2024/9/24 10:28 
# @Author : DirtyBoy 
# @File : train_correction_model.py
import argparse, random
from myexperiments.utils import correct_model_training_data_processing, save_correction_model, load_correction_model
from sklearn import svm

parser = argparse.ArgumentParser()
parser.add_argument('-test_data_type', '-tdt', type=str, default="correct")
parser.add_argument('-data_type', '-dt', type=str, default="base")
parser.add_argument('-feature_type', '-ft', type=str, default="drebin1")
args = parser.parse_args()
feature_type = args.feature_type
data_type = args.data_type
test_data_type = args.test_data_type

feature, true_index, false_index = correct_model_training_data_processing(feature_type, data_type, test_data_type)
true_index = random.sample(true_index, len(false_index))
feature = feature[true_index + false_index]
label = [0] * len(true_index) + [1] * len(false_index)

print(feature.shape)
clf = svm.SVC(C=1000, gamma=0.01, kernel="linear")

clf.fit(X=feature, y=label)

save_correction_model(clf, 'correction_model/' + feature_type + '_' + data_type + '_correct_' + test_data_type + '.pkl')
