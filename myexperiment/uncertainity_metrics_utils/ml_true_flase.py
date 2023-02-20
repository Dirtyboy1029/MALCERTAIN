# -*- coding: utf-8 -*- 
# @Time : 2022/9/26 16:02 
# @Author : DirtyBoy 
# @File : ml_true_flase.py
from sklearn.utils import shuffle
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import joblib
from sklearn import svm  # svm支持向量机
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from myexperiment.ML_acc.utils import get_data_frame, train_test_split_data_frame, feature_selection, evaluate_acc, \
    evaluate_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-experiment_type', type=str, default="train", choices=['train', 'test'])
    parser.add_argument('-save_model', type=str, default="n", choices=['y', 'n'])
    parser.add_argument('-data_type', type=str, default="small_drebin")
    parser.add_argument('-test_model_type', type=str, default="small_drebin",
                        choices=['small_drebin', 'small_androzoo', 'big_androzoo'])
    args = parser.parse_args()
    experiment_type = args.experiment_type
    data_type = args.data_type
    if args.save_model == "y":
        save_model = True
    elif args.save_model == "n":
        save_model = False
    data_base_path = 'D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\output\\uc_csv-'
    data_base_path = data_base_path + data_type
    data = get_data_frame(data_base_path)
    clf_svm = svm.SVC(C=1000, gamma=0.01, kernel="linear")
    clf_knn = KNeighborsClassifier(n_neighbors=8, p=1, weights="distance")
    clf_dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=6, criterion="gini")
    clf_RF = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=10)

    if experiment_type == "train":
        x_train, x_test, y_train, y_test, y_train_true, y_test_true, y_train_pred, y_test_pred = train_test_split_data_frame(
            data)
        result_svm = evaluate_model(clf_svm, x_train, x_test, y_train, y_test, type="svm_" + data_type,
                                    save_model=save_model)
        result_knn = evaluate_model(clf_knn, x_train, x_test, y_train, y_test, type="knn_" + data_type,
                                    save_model=save_model)
        result_dt = evaluate_model(clf_dt, x_train, x_test, y_train, y_test, type="dt_" + data_type,
                                   save_model=save_model)
        result_rf = evaluate_model(clf_RF, x_train, x_test, y_train, y_test, type="rf_" + data_type,
                                   save_model=save_model)

        print("---------------------------------svm--------------------------------------")
        evaluate_acc(result_svm, y_test, y_test_true, y_test_pred)
        print("---------------------------------knn--------------------------------------")
        evaluate_acc(result_knn, y_test, y_test_true, y_test_pred)
        print("---------------------------------dt--------------------------------------")
        evaluate_acc(result_dt, y_test, y_test_true, y_test_pred)
        print("---------------------------------rf--------------------------------------")
        evaluate_acc(result_rf, y_test, y_test_true, y_test_pred)

    elif experiment_type == "test":
        test_model_type = args.test_model_type
        x_test, y_test, y_test_true, y_test_pred = train_test_split_data_frame(data, is_test=True)

        clf_svm = joblib.load('/home/lhd/uncertainity_malware/myexperiment/uncertainity_metrics_utils/model/svm_' + test_model_type + '.pkl')
        clf_knn = joblib.load('/home/lhd/uncertainity_malware/myexperiment/uncertainity_metrics_utils/model/knn_' + test_model_type + '.pkl')
        clf_dt = joblib.load('/home/lhd/uncertainity_malware/myexperiment/uncertainity_metrics_utils/model/dt_' + test_model_type + '.pkl')
        clf_RF = joblib.load('/home/lhd/uncertainity_malware/myexperiment/uncertainity_metrics_utils/model/rf_' + test_model_type + '.pkl')

        result_svm = clf_svm.predict(x_test)
        result_knn = clf_knn.predict(x_test)
        result_dt = clf_dt.predict(x_test)
        result_rf = clf_RF.predict(x_test)

        print("---------------------------------svm--------------------------------------")
        evaluate_acc(result_svm, y_test, y_test_true, y_test_pred)
        print("---------------------------------knn--------------------------------------")
        evaluate_acc(result_knn, y_test, y_test_true, y_test_pred)
        print("---------------------------------dt--------------------------------------")
        evaluate_acc(result_dt, y_test, y_test_true, y_test_pred)
        print("---------------------------------rf--------------------------------------")
        evaluate_acc(result_rf, y_test, y_test_true, y_test_pred)
