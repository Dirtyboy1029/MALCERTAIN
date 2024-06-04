# -*- coding: utf-8 -*- 

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
from myexperiment.uncertainty_metrics_utils.ml_utils import get_data_frame, train_test_split_data_frame, feature_selection, evaluate_acc, \
    evaluate_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-experiment_type', "-e", type=str, default="train", choices=['train', 'test'])
    parser.add_argument('-train_data_size', "-size", type=float, default="1.0")
    parser.add_argument('-feature_num',"-fn", type=int, default=80)
    parser.add_argument('-save_model', "-s", type=str, default="n", choices=['y', 'n'])
    parser.add_argument('-balance', "-b", type=str, default="n", choices=['y', 'n'])
    parser.add_argument('-feature_select', "-fs", type=str, default="n", choices=['y', 'n'])
    parser.add_argument('-data_type', type=str, default="small_drebin")
    parser.add_argument('-test_model_type', type=str, default="small_drebin_10010")
    args = parser.parse_args()
    experiment_type = args.experiment_type
    data_type = args.data_type
    train_data_size = args.train_data_size
    feature_num = args.feature_num
    if args.balance == "y":
        PN = True
        is_balance = "balance"
    elif args.balance == "n":
        PN = False
        is_balance = "not_balance"
    is_feature_select = False
    if args.feature_select == "y":
        is_feature_select = True
    elif args.feature_select == "n":
        is_feature_select = False

    if args.save_model == "y":
        save_model = True
    elif args.save_model == "n":
        save_model = False
    data_base_path = 'D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\output\\uc_csv-'
    data_base_path = data_base_path + data_type
    data = get_data_frame(data_base_path)
    clf_svm = svm.SVC(C=1000, gamma=0.01, kernel="linear")
    clf_knn = KNeighborsClassifier(n_neighbors=6, p=1, weights="distance")
    clf_dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=6, criterion="gini")
    clf_RF = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=6)

    if experiment_type == "train":
        x_train,  y_train,  y_train_true,  y_train_pred= train_test_split_data_frame(
            data, train_data_size=train_data_size, PN=PN, is_balance=is_balance)
        if is_feature_select:
            x_train = feature_selection(x_train, y_train,
                                                type=data_type + "_" + is_balance + "_" + str(train_data_size) + "_" + str(
                                                    feature_num),
                                                feature_num=feature_num)
        result_svm = evaluate_model(clf_svm, x_train,  y_train,
                                    type="svm_" + data_type + "_" + is_balance + "_" + str(train_data_size),
                                    save_model=save_model, PN=PN, feature_select=is_feature_select,
                                    feature_num=feature_num)
        result_knn = evaluate_model(clf_knn, x_train,  y_train,
                                    type="knn_" + data_type + "_" + is_balance + "_" + str(train_data_size),
                                    save_model=save_model, PN=PN, feature_select=is_feature_select,
                                    feature_num=feature_num)
        result_dt = evaluate_model(clf_dt, x_train,  y_train,
                                   type="dt_" + data_type + "_" + is_balance + "_" + str(train_data_size),
                                   save_model=save_model, PN=PN, feature_select=is_feature_select,
                                   feature_num=feature_num)
        result_rf = evaluate_model(clf_RF, x_train,  y_train,
                                   type="rf_" + data_type + "_" + is_balance + "_" + str(train_data_size),
                                   save_model=save_model, PN=PN, feature_select=is_feature_select,
                                   feature_num=feature_num)

        # print("---------------------------------svm--------------------------------------")
        # evaluate_acc(result_svm, y_test, y_test_true, y_test_pred)
        # print("---------------------------------knn--------------------------------------")
        # evaluate_acc(result_knn, y_test, y_test_true, y_test_pred)
        # print("---------------------------------dt--------------------------------------")
        # evaluate_acc(result_dt, y_test, y_test_true, y_test_pred)
        # print("---------------------------------rf--------------------------------------")
        # evaluate_acc(result_rf, y_test, y_test_true, y_test_pred)

    elif experiment_type == "test":
        test_model_type = args.test_model_type
        type_list = test_model_type.split("_")
        type = type_list[0] + "_" + type_list[1]
        if PN:
            clf_svm = joblib.load(
                'D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\old_PN\\svm_' + test_model_type + '.pkl')
            clf_knn = joblib.load(
                'D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\old_PN\\knn_' + test_model_type + '.pkl')
            clf_dt = joblib.load(
                'D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\old_PN\\dt_' + test_model_type + '.pkl')
            clf_RF = joblib.load(
                'D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\old_PN\\rf_' + test_model_type + '.pkl')
        else:
            clf_svm = joblib.load(
                'D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\old_not_PN\\svm_' + test_model_type + '.pkl')
            clf_knn = joblib.load(
                'D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\old_not_PN\\knn_' + test_model_type + '.pkl')
            clf_dt = joblib.load(
                'D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\old_not_PN\\dt_' + test_model_type + '.pkl')
            clf_RF = joblib.load(
                'D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\old_not_PN\\rf_' + test_model_type + '.pkl')
        if data_type == "small_at_adma" or data_type == "small_d_ade_ma" \
                or data_type == "small_at_ma" or data_type == "small_ade_ma" \
                or data_type == "small_basic_dnn" or data_type == "small_at_rfgsm" \
                or data_type == "multi_at_ma" or data_type == "multi_ade_ma" \
                or data_type == "multi_basic_dnn" or data_type == "multi_at_rfgsm":
            x_test_adv, y_test_adv, y_test_true_adv, y_test_pred_adv, x_test_pst, y_test_pst, y_test_true_pst, y_test_pred_pst = train_test_split_data_frame(
                data, is_test=True,is_balance=is_balance,
                is_adv=True, data_type=type,feature_num=feature_num)
            result_svm_adv = clf_svm.predict(x_test_adv)
            result_knn_adv = clf_knn.predict(x_test_adv)
            result_dt_adv = clf_dt.predict(x_test_adv)
            result_rf_adv = clf_RF.predict(x_test_adv)

            result_svm_pst = clf_svm.predict(x_test_pst)
            result_knn_pst = clf_knn.predict(x_test_pst)
            result_dt_pst = clf_dt.predict(x_test_pst)
            result_rf_pst = clf_RF.predict(x_test_pst)

            print("---------------------------------svm-adv--------------------------------------")
            evaluate_acc(result_svm_adv, y_test_adv, y_test_true_adv, y_test_pred_adv)
            print("---------------------------------svm-pst--------------------------------------")
            evaluate_acc(result_svm_pst, y_test_pst, y_test_true_pst, y_test_pred_pst)
            print("---------------------------------knn-adv--------------------------------------")
            evaluate_acc(result_knn_adv, y_test_adv, y_test_true_adv, y_test_pred_adv)
            print("---------------------------------knn-pst--------------------------------------")
            evaluate_acc(result_knn_pst, y_test_pst, y_test_true_pst, y_test_pred_pst)
            print("---------------------------------dt-adv--------------------------------------")
            evaluate_acc(result_dt_adv, y_test_adv, y_test_true_adv, y_test_pred_adv)
            print("---------------------------------dt-pst--------------------------------------")
            evaluate_acc(result_dt_pst, y_test_pst, y_test_true_pst, y_test_pred_pst)
            print("---------------------------------rf-adv--------------------------------------")
            evaluate_acc(result_rf_adv, y_test_adv, y_test_true_adv, y_test_pred_adv)
            print("---------------------------------rf-pst--------------------------------------")
            evaluate_acc(result_rf_pst, y_test_pst, y_test_true_pst, y_test_pred_pst)
        else:
            x_test, y_test, y_test_true, y_test_pred = train_test_split_data_frame(
                data, is_test=True,is_balance=is_balance,
                data_type=type,feature_num=feature_num)
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
