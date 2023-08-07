# -*- coding: utf-8 -*- 

# @File : tools.py
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import tensorflow as tf
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import seaborn
from core.post_calibration.temperature_scaling import inverse_sigmoid,find_scaling_temperature
import math
from collections import Counter
import os
import joblib
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2

def get_temperature(model,data,label):
    prob = np.squeeze(model.predict(data, use_prob=True))
    logits = inverse_sigmoid(prob)
    temperature = find_scaling_temperature(label, logits)
    return temperature

def predictive_entropy(DataList):
    '''
        计算随机变量 DataList 的熵
    '''
    counts = len(DataList)  # 总数量
    counter = Counter(DataList)  # 每个变量出现的次数
    prob = {i[0]: i[1] / counts for i in counter.items()}  # 计算每个变量的 p*log(p)
    H = - sum([i[1] * math.log2(i[1]) for i in prob.items()])  # 计算熵

    return H

def predictive_entropy1(predictions):
    epsilon = sys.float_info.min
    predictive_entropy = -np.sum(np.mean(predictions, axis=0) * np.log(np.mean(predictions, axis=0) + epsilon),
                                 axis=-1)
    return predictive_entropy

def Save_to_Csv(data, file_name, Save_format='csv', Save_type='col'):
    # data
    # 输入为一个字典，格式： { '列名称': 数据,....}
    # 列名即为CSV中数据对应的列名， 数据为一个列表
    # file_name 存储文件的名字
    # Save_format 为存储类型， 默认csv格式， 可改为 excel
    # Save_type 存储类型 默认按列存储， 否则按行存储
    # 默认存储在当前路径下
    Name = []
    times = 0
    if Save_type == 'col':
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List).reshape(-1, 1)
            else:
                Data = np.hstack((Data, np.array(List).reshape(-1, 1)))
            times += 1
        Pd_data = pd.DataFrame(columns=Name, data=Data)
    else:
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List)
            else:
                Data = np.vstack((Data, np.array(List)))
            times += 1
        Pd_data = pd.DataFrame(index=Name, data=Data)

    if Save_format == 'csv':
        Pd_data.to_csv('/home/lhd/uncertainity_malware/myexperiment/output/adv_uc_csv/' + file_name + '.csv', encoding='utf-8')
    else:
        Pd_data.to_excel('/home/lhd/uncertainity_malware/myexperiment/output/adv_uc_csv/' + file_name + '.xls', encoding='utf-8')

def get_cdf_png(pst_path,adv_path,title,png_name,calibration=False,true_flase=False):
    if true_flase:
        list_adv_true = pd.read_csv(pst_path)['1'].to_list()
        list_adv_flase = pd.read_csv(adv_path)['1'].to_list()
    else:
        list_adv_true = np.loadtxt(pst_path)
        list_adv_flase = np.loadtxt(adv_path)

    L_list = [np.max(list_adv_true), np.max(list_adv_flase)]
    idx = L_list.index(max(L_list))

    print(L_list)
    print(idx)

    res_true = stats.relfreq(list_adv_true, numbins=100)
    res_flase = stats.relfreq(list_adv_flase, numbins=100)

    if (idx == 0):
        res = res_true
    elif (idx == 1):
        res = res_flase

    x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
    #plt.figure(figsize=(5, 4))
    y_true = np.cumsum(res_true.frequency)
    y_flase = np.cumsum(res_flase.frequency)
    if not calibration:
        name = "D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\output\\cdf_png\\uc\\un_Calibration\\" + png_name + "_cdf.png"
    else:
        name = "D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\output\\cdf_png\\uc\\Calibration\\" + png_name + "_cdf.png"
        title = title + "_calibration"
    plt.figure()
    if true_flase:
        plt.plot(x, y_true, c="r", label="true")
        plt.plot(x, y_flase, c="b", label="flase")
    else:
        plt.plot(x, y_true, c="r", label="prist")
        plt.plot(x, y_flase, c="b", label="adv")
    plt.title(title)
    plt.legend()

    plt.savefig(name)

def metrics2csv():
    pass

def get_data_frame(base_folder, is_flase=False):
    df_base_true = pd.read_csv(os.path.join(base_folder, "label/pred_true.csv"))
    df_base_flase = pd.read_csv(os.path.join(base_folder, "label/pred_flase.csv"))

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


def train_test_split_data_frame(df, is_test=False, train_data_size=1, PN=False,is_adv = False):
    df = df.sample(int(df.shape[0] * train_data_size))
    data_shuffle = shuffle(df)
    data_0 = data_shuffle[data_shuffle.label == 0]
    data_1 = data_shuffle[data_shuffle.label == 1]
    if PN:
        data_1 = data_1.sample(data_0.shape[0])
    data_ = pd.concat([data_0, data_1])
    data_shuffle = shuffle(data_)
    print("P smaple num =", data_1.shape[0])
    print("all smaple num =", data_1.shape[0] + data_0.shape[0])
    if not is_test:
        train_data, test_data = train_test_split(data_shuffle, test_size=0.3)
        train_data = train_data.iloc[:, 1:]
        test_data = test_data.iloc[:, 1:]
        y_train = train_data.iloc[:, :1]
        y_test = test_data.iloc[:, :1]
        y_train_pred = train_data.iloc[:, 2:3]
        y_test_pred = test_data.iloc[:, 2:3]
        y_train_true = train_data.iloc[:, 1:2]
        y_test_true = test_data.iloc[:, 1:2]
        x_train = train_data.iloc[:, 3:]
        x_test = test_data.iloc[:, 3:]
        return x_train, x_test, y_train, y_test, y_train_true, y_test_true, y_train_pred, y_test_pred
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
            return x_test_adv,y_test_adv,y_test_true_adv,y_test_pred_adv ,\
                   x_test_pst, y_test_pst, y_test_true_pst, y_test_pred_pst
        else:
            test_data = data_shuffle.iloc[:, 1:]
            y_test = test_data.iloc[:, :1]
            y_test_pred = test_data.iloc[:, 2:3]
            x_test = test_data.iloc[:, 3:]
            y_test_true = test_data.iloc[:, 1:2]
            return x_test, y_test, y_test_true, y_test_pred


def feature_selection(x_train, y_train, x_test, feature_num):
    selector = SelectKBest(f_classif, k=feature_num)
    new = selector.fit(x_train, y_train).get_support()
    num = []
    for i in range(len(new)):
        if new[i]:
            num.append(i)
    x_train = x_train.iloc[:, num]
    x_test = x_test.iloc[:, num]
    return x_train, x_test


def evaluate_model(clf, x_train, x_test, y_train, y_test, type, save_model=False, feature_select=False, feature_num=80,
                   PN=False):
    if feature_select:
        x_train, x_test = feature_selection(x_train, y_train, x_test, feature_num=feature_num)
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
                joblib.dump(clf, "D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\PN\\" + type + '.pkl')
            else:
                joblib.dump(clf, "D:\\Pycharm\\Project\\malware-uncertainty\\myexperiment\\ml_model\\not_PN\\" + type + '.pkl')
    result = clf.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, result).ravel()
    fpr = fp / float(tn + fp)
    fnr = fn / float(tp + fn)
    f1 = f1_score(y_test, result, average='binary')
    print("---------------------" + type + "-------------------------")
    print("False Negative Rate (FNR) is " + str(fnr * 100) + "%, False Positive Rate (FPR) is " + str(
        fpr * 100) + "%, F1 score is " + str(f1 * 100) + "%,acc is " + str(clf.score(x_test, y_test) * 100) + "%")
    return result


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
    print("矫正FNR准确率：", accuracy_score(y_test_true, y_new))
    print("矫正准确率：", accuracy_score(y_test_true, y_new_))
