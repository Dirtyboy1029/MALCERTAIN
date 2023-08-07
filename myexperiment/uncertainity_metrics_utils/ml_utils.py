from sklearn.utils import shuffle

import os
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_classif, chi2
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


def get_data_frame(base_folder, is_flase=False):
    df_base_true = pd.read_csv(os.path.join(base_folder, "label\pred_true.csv"))
    df_base_flase = pd.read_csv(os.path.join(base_folder, "label\pred_flase.csv"))

    for root, dirs, files in os.walk(os.path.join(base_folder, "all")):
        for item in files:
            df_tmp = pd.read_csv(os.path.join(root, item))
            df_tmp.rename(columns={'0': 'apk_name'}, inplace=True)
            df_tmp.rename(columns={'1': os.path.splitext(item)[0]}, inplace=True)
            del df_tmp['Unnamed: 0']
            df_base_true = pd.merge(df_base_true, df_tmp, on='apk_name')
            df_base_flase = pd.merge(df_base_flase, df_tmp, on='apk_name')

    del df_base_flase['Unnamed: 0']
    del df_base_true['Unnamed: 0']
    if not is_flase:
        data = pd.concat([df_base_flase, df_base_true], axis=0)
    else:
        data = df_base_flase
    return data


def train_test_split_data_frame(df, is_test=False, train_data_size=1.0, PN=False, is_adv=False, is_balance="balance",
                                data_type="small_drebin", feature_num=80):
    df = df.sample(int(df.shape[0] * train_data_size))
    data_shuffle = shuffle(df)
    csv_name = is_balance + "_" + str(train_data_size) + ".csv"

    data_0 = data_shuffle[data_shuffle.label == 0]
    data_1 = data_shuffle[data_shuffle.label == 1]
    if PN:
        data_1 = data_1.sample(data_0.shape[0])
    data_ = pd.concat([data_0, data_1])
    data_shuffle = shuffle(data_)
    data_shuffle.to_csv(csv_name)
    print("P smaple num =", data_1.shape[0])
    print("all smaple num =", data_1.shape[0] + data_0.shape[0])
    if not is_test:
        #train_data, test_data = train_test_split(data_shuffle, test_size=0.3)
        train_data = data_shuffle
        # train_data['apk_name'].to_csv(
        #     "D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\data\\" + is_balance + "_" + str(
        #         train_data_size) + ".csv")
        train_data = train_data.iloc[:, 1:]
        #test_data = test_data.iloc[:, 1:]
        y_train = train_data.iloc[:, :1]
        #y_test = test_data.iloc[:, :1]
        y_train_pred = train_data.iloc[:, 2:3]
        #y_test_pred = test_data.iloc[:, 2:3]
        y_train_true = train_data.iloc[:, 1:2]
        #y_test_true = test_data.iloc[:, 1:2]
        x_train = train_data.iloc[:, 3:]
        #x_test = test_data.iloc[:, 3:]
        return x_train,  y_train,  y_train_true,  y_train_pred
    else:
        if is_adv:
            test_data_pst = data_shuffle[data_shuffle["apk_name"].str.contains("_pst")]
            test_data_adv = data_shuffle[data_shuffle["apk_name"].str.contains("_adv")]
            test_data_pst = test_data_pst.iloc[:, 1:]
            test_data_adv = test_data_adv.iloc[:, 1:]
            y_test_pst = test_data_pst.iloc[:, :1]
            y_test_pred_pst = test_data_pst.iloc[:, 2:3]
            x_test_pst = test_data_pst.iloc[:, 3:]
            y_test_true_pst = test_data_pst.iloc[:, 1:2]
            y_test_adv = test_data_adv.iloc[:, :1]
            y_test_pred_adv = test_data_adv.iloc[:, 2:3]
            x_test_adv = test_data_adv.iloc[:, 3:]
            y_test_true_adv = test_data_adv.iloc[:, 1:2]
            return x_test_adv, y_test_adv, y_test_true_adv, y_test_pred_adv, \
                   x_test_pst, y_test_pst, y_test_true_pst, y_test_pred_pst
        else:
            # with open(
            #         "D:\\Pycharm\Project\\malware-uncertainty\\myexperiment\\ml_model\\feature_select_file\\" + data_type + "_" + is_balance + "_" + str(
            #             train_data_size) + "_" + str(feature_num)) as f:
            #     name = f.read().splitlines()
            # print("read file:" + data_type + "_" + is_balance + "_" + str(
            #     train_data_size) + "_" + str(feature_num))
            test_data = data_shuffle.iloc[:, 1:]
            y_test = test_data.iloc[:, :1]
            y_test_pred = test_data.iloc[:, 2:3]
            x_test = test_data.iloc[:, 3:]
            y_test_true = test_data.iloc[:, 1:2]
            return x_test, y_test, y_test_true, y_test_pred


def feature_selection(x_train, y_train,  type, feature_num):
    selector = SelectKBest(f_classif, k=feature_num)
    new = selector.fit(x_train, y_train).get_support()
    num = []
    for i in range(len(new)):
        if new[i]:
            num.append(i)
    x_train = x_train.iloc[:, num]
    #x_test = x_test.iloc[:, num]
    with open("D:\\Pycharm\Project\\malware-uncertainty\\myexperiment\\ml_model\\feature_select_file\\" + type,
              "w") as f:
        for item in list(x_train.columns):
            f.write(item + "\n")
    return x_train#, x_test


def evaluate_model(clf, x_train,  y_train, type, save_model=False, feature_select=False, feature_num=80,
                   PN=False):
    if feature_select:

        clf.fit(X=x_train, y=y_train)
        if save_model:
            if PN:
                joblib.dump(clf,
                            "D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\PN\\" + type + '_' + str(
                                feature_num) + '_' + '.pkl')
            else:
                joblib.dump(clf,
                            "D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\not_PN\\" + type + '_' + str(
                                feature_num) + '_' + '.pkl')
    else:
        clf.fit(X=x_train, y=y_train)
        if save_model:
            if PN:
                joblib.dump(clf,
                            "D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\PN\\" + type + '.pkl')
            else:
                joblib.dump(clf,
                            "D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\not_PN\\" + type + '.pkl')
    #result = clf.predict(x_test)
    # tn, fp, fn, tp = confusion_matrix(y_test, result).ravel()
    # fpr = fp / float(tn + fp)
    # fnr = fn / float(tp + fn)
    # f1 = f1_score(y_test, result, average='binary')
    # print("---------------------" + type + "-------------------------")
    # print("False Negative Rate (FNR) is " + str(fnr * 100) + "%, False Positive Rate (FPR) is " + str(
    #     fpr * 100) + "%, F1 score is " + str(f1 * 100) + "%,acc is " + str(clf.score(x_test, y_test) * 100) + "%")
    #return result

def confusion(y_test_true, y_new):
    tn, fp, fn, tp = confusion_matrix(y_test_true, y_new).ravel()
    fpr = fp / float(tn + fp)
    fnr = fn / float(tp + fn)
    f1 = f1_score(y_test_true, y_new, average='binary')

    print("False Negative Rate (FNR) is " + str(fnr * 100) + "%, False Positive Rate (FPR) is " + str(
        fpr * 100) + "%, F1 score is " + str(f1 * 100) + "%")

def evaluate_acc(result, y_test, y_test_true, y_test_pred):
    y_new = []
    y_new_ = []
    for i in range(len(y_test)):
        if int(result[i]) == 1:
            y_new.append(np.array(y_test_pred).tolist()[i][0])
        elif int(result[i]) == 0 and int(np.array(y_test_pred).tolist()[i][0]) == 0:
            y_new.append(int(not int(np.array(y_test_pred).tolist()[i][0])))
        else:
            y_new.append(int(np.array(y_test_pred).tolist()[i][0]))
    for i in range(len(y_test)):
        if int(result[i]) == 1:
            y_new_.append(np.array(y_test_pred).tolist()[i][0])

        else:
            y_new_.append(not int(np.array(y_test_pred).tolist()[i][0]))

    print("原本准确率：", accuracy_score(y_test_true, y_test_pred))
    confusion(y_test_true, y_test_pred)
    print("矫正FNR准确率：", accuracy_score(y_test_true, y_new))
    confusion(y_test_true, y_new)

    print("矫正准确率：", accuracy_score(y_test_true, y_new_))
    confusion(y_test_true, y_new_)
