# -*- coding: utf-8 -*- 
# @Time : 2022/10/7 21:50 
# @Author : DirtyBoy 
# @File : utils.py

from core.ensemble.model_hp import train_hparam, mc_dropout_hparam, bayesian_ensemble_hparam
from experiments.adv import data_preprocessing_get_name
from core.ensemble.deep_ensemble import WeightedDeepEnsemble
from core.ensemble.mc_dropout import MCDropout
from core.ensemble.bayesian_ensemble import BayesianEnsemble
from core.ensemble.deep_ensemble import DeepEnsemble
from core.ensemble.vanilla import Vanilla
from tools.metrics import predictive_std, predictive_kld
from tools.metrics import entropy as predictive_entropy
# from myexperiment.uncertainity_metrics_utils.tools import predictive_entropy
import numpy as np
from myexperiment.uncertainity_metrics_utils.tools import Save_to_Csv, get_temperature
from core.post_calibration.temperature_scaling import apply_temperature_scaling
import os
from tools import utils
import pandas as pd


def get_label(model_path, ood_data, oos_y, oos_filenames, architecture_type):
    base_model = Vanilla(architecture_type=architecture_type, model_directory=model_path)

    pred_prob = base_model.predict(ood_data, use_prob=True)
    pred_class = []
    pred_true_name = []
    pred_flase_name = []
    pred_true_y_really = []
    pred_true_y = []
    pred_flase_y_really = []
    pred_flase_y = []
    for i in range(len(oos_filenames)):
        if pred_prob[i] > 0.5:
            pred_class.append(1)
        else:
            pred_class.append(0)

    for i in range(len(oos_filenames)):
        if pred_class[i] == oos_y[i]:
            pred_true_name.append(oos_filenames[i])
            pred_true_y_really.append(oos_y[i])
            pred_true_y.append(pred_class[i])
        else:
            pred_flase_name.append(oos_filenames[i])
            pred_flase_y_really.append(oos_y[i])
            pred_flase_y.append(pred_class[i])

    print("总样本数:", len(oos_filenames))
    print("恶意样本数:", np.sum(oos_y))
    print("所有样本预测正确：", len(pred_true_name))
    print("所有样本预测错误：", len(pred_flase_name))

    df_prob = pd.DataFrame()
    df_prob["apk_name"] = oos_filenames
    df_prob["prob"] = pred_prob

    df_true = pd.DataFrame()
    df_true["apk_name"] = pred_true_name
    df_true["label"] = [1.0] * len(pred_true_name)
    df_true["really_label"] = np.array(pred_true_y_really)
    df_true["pred_label"] = np.array(pred_true_y)
    df_flase = pd.DataFrame()
    df_flase["apk_name"] = pred_flase_name
    df_flase["label"] = [0.0] * len(pred_flase_name)
    df_flase["really_label"] = np.array(pred_flase_y_really)
    df_flase["pred_label"] = np.array(pred_flase_y)
    df_true.to_csv("/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/label/pred_true.csv")
    df_flase.to_csv(
        "/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/label/pred_flase.csv")
    df_prob.to_csv(
        "/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/pred_prob.csv")

    return pred_true_name, pred_flase_name


def epoch_ensemble_uc(base_path, ood_data, oos_y, oos_filenames, architecture_type):
    csv_save_path = "/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/"
    model_name_list = os.listdir(base_path)
    model_path_list = [os.path.join(base_path, item) for item in model_name_list]
    model_list = [Vanilla(architecture_type=architecture_type,
                          model_directory=item) for item in model_path_list]

    pred_ = [item.predict(ood_data, use_prob=True) for item in model_list]

    save_std_path_true = csv_save_path + "std/epoch_ensemble_std_true.csv"
    save_std_path_flase = csv_save_path + "std/epoch_ensemble_std_flase.csv"
    save_ent_path_true = csv_save_path + "entropy/epoch_ensemble_ent_true.csv"
    save_ent_path_flase = csv_save_path + "entropy/epoch_ensemble_ent_flase.csv"
    save_kld_path_true = csv_save_path + "kld/epoch_ensemble_kld_true.csv"
    save_kld_path_flase = csv_save_path + "kld/epoch_ensemble_kld_flase.csv"

    entropy = []
    kld = []
    std = []
    pred_prob = []

    for i in range(len(oos_filenames)):
        tmp = []
        for item in pred_:
            tmp.append(item[i][0])
        pred_prob.append(np.mean(tmp))
        std.append(predictive_std(tmp, number=len(model_list))[0])
        entropy.append(predictive_entropy(tmp))
        kld.append(predictive_kld(tmp, number=len(model_list))[0])

    pred_class = []
    entropy_true = []
    entropy_flase = []
    kld_true = []
    kld_flase = []
    std_true = []
    std_flase = []
    for i in range(len(oos_filenames)):
        if pred_prob[i] > 0.5:
            pred_class.append(1)
        else:
            pred_class.append(0)

    for i in range(len(oos_filenames)):
        if int(pred_class[i]) == int(oos_y[i]):
            entropy_true.append([oos_filenames[i], entropy[i]])
            kld_true.append([oos_filenames[i], kld[i][0]])
            std_true.append([oos_filenames[i], std[i][0]])
        else:
            entropy_flase.append([oos_filenames[i], entropy[i]])
            kld_flase.append([oos_filenames[i], kld[i][0]])
            std_flase.append([oos_filenames[i], std[i][0]])

    pd.DataFrame(entropy_true).to_csv(save_ent_path_true)
    pd.DataFrame(entropy_flase).to_csv(save_ent_path_flase)
    pd.DataFrame(kld_true).to_csv(save_kld_path_true)
    pd.DataFrame(kld_flase).to_csv(save_kld_path_flase)
    pd.DataFrame(std_true).to_csv(save_std_path_true)
    pd.DataFrame(std_flase).to_csv(save_std_path_flase)

    save_std_path_all = csv_save_path + "all/epoch_ensemble_std.csv"
    save_ent_path_all = csv_save_path + "all/epoch_ensemble_ent.csv"
    save_kld_path_all = csv_save_path + "all/epoch_ensemble_kld.csv"
    pd.DataFrame(entropy_true + entropy_flase).to_csv(save_ent_path_all)
    pd.DataFrame(kld_true + kld_flase).to_csv(save_kld_path_all)
    pd.DataFrame(std_true + std_flase).to_csv(save_std_path_all)

    return entropy_true, entropy_flase, kld_true, kld_flase, std_true, std_flase


def get_bay_uc(model_directory, ood_data, oos_y, oos_filenames, epoch, architecture_type):
    csv_save_path = "/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/"
    bay = BayesianEnsemble(architecture_type=architecture_type, model_directory=model_directory)
    bay.hparam = utils.merge_namedtuples(train_hparam, bayesian_ensemble_hparam)
    bay_pred = bay.predict(ood_data)

    entropy = []
    kld = []
    std = []
    pred_prob = []

    med_min = []
    max_med = []
    med_mean = []
    max_max2 = []
    max_mean = []
    mean_min = []

    for item in bay_pred:
        end_list = []
        for i in range(bay.hparam.n_sampling):
            end_list.append(item[i][0])
        max2 = np.sort(end_list)[-2]
        mean = np.mean(end_list)
        med = np.median(end_list)
        max = np.max(end_list)
        min = np.min(end_list)

        med_mean.append(med - mean)
        max_med.append(max - med)
        max_max2.append(max - max2)
        med_min.append(med - min)
        max_mean.append(max - mean)
        mean_min.append(mean - min)

        pred_prob.append(mean)
        entropy.append(predictive_entropy(end_list))
        kld.append(predictive_kld(end_list, number=bay.hparam.n_sampling)[0])
        std.append(predictive_std(end_list, number=bay.hparam.n_sampling)[0])

    pred_class = []
    entropy_true = []
    entropy_flase = []
    kld_true = []
    kld_flase = []
    std_true = []
    std_flase = []

    med_min_list_flase = []
    max_med_list_flase = []
    med_min_list_true = []
    max_med_list_true = []
    med_mean_list_true = []
    med_mean_list_flase = []
    max_max2_list_true = []
    max_max2_list_flase = []
    max_mean_list_true = []
    max_mean_list_flase = []
    mean_min_list_true = []
    mean_min_list_flase = []

    for i in range(len(oos_filenames)):
        if pred_prob[i] > 0.5:
            pred_class.append(1)
        else:
            pred_class.append(0)

    for i in range(len(oos_filenames)):
        if pred_class[i] == oos_y[i]:
            med_mean_list_true.append([oos_filenames[i], med_mean[i]])
            max_med_list_true.append([oos_filenames[i], max_med[i]])
            max_max2_list_true.append([oos_filenames[i], max_max2[i]])
            med_min_list_true.append([oos_filenames[i], med_min[i]])
            max_mean_list_true.append([oos_filenames[i], max_mean[i]])
            mean_min_list_true.append([oos_filenames[i], mean_min[i]])

            entropy_true.append([oos_filenames[i], entropy[i]])
            kld_true.append([oos_filenames[i], kld[i][0]])
            std_true.append([oos_filenames[i], std[i][0]])
        else:
            med_mean_list_flase.append([oos_filenames[i], med_mean[i]])
            max_med_list_flase.append([oos_filenames[i], max_med[i]])
            max_max2_list_flase.append([oos_filenames[i], max_max2[i]])
            med_min_list_flase.append([oos_filenames[i], med_min[i]])
            max_mean_list_flase.append([oos_filenames[i], max_mean[i]])
            mean_min_list_flase.append([oos_filenames[i], mean_min[i]])

            entropy_flase.append([oos_filenames[i], entropy[i]])
            kld_flase.append([oos_filenames[i], kld[i][0]])
            std_flase.append([oos_filenames[i], std[i][0]])

    save_std_path_true = csv_save_path + "std/true_bayesian_std_" + str(epoch) + ".csv"
    save_std_path_flase = csv_save_path + "std/flase_bayesian_std_" + str(epoch) + ".csv"
    save_ent_path_true = csv_save_path + "entropy/true_bayesian_entropy_" + str(epoch) + ".csv"
    save_ent_path_flase = csv_save_path + "entropy/flase_bayesian_entropy_" + str(epoch) + ".csv"
    save_kld_path_true = csv_save_path + "kld/true_bayesian_kld_" + str(epoch) + ".csv"
    save_kld_path_flase = csv_save_path + "kld/flase_bayesian_kld_" + str(epoch) + ".csv"

    pd.DataFrame(entropy_true).to_csv(save_ent_path_true)
    pd.DataFrame(entropy_flase).to_csv(save_ent_path_flase)
    pd.DataFrame(kld_true).to_csv(save_kld_path_true)
    pd.DataFrame(kld_flase).to_csv(save_kld_path_flase)
    pd.DataFrame(std_true).to_csv(save_std_path_true)
    pd.DataFrame(std_flase).to_csv(save_std_path_flase)

    save_med_min_true = csv_save_path + "med_min/true_bayesian_med_min_" + str(epoch) + ".csv"
    save_max_med_true = csv_save_path + "max_med/true_bayesian_max_med_" + str(epoch) + ".csv"
    save_med_mean_true = csv_save_path + "med_mean/true_bayesian_med_mean_" + str(epoch) + ".csv"
    save_max_max2_true = csv_save_path + "max_max2/true_bayesian_max_max2_" + str(epoch) + ".csv"
    save_max_mean_true = csv_save_path + "max_mean/true_bayesian_max_mean_" + str(epoch) + ".csv"
    save_mean_min_true = csv_save_path + "mean_min/true_bayesian_mean_min_" + str(epoch) + ".csv"
    save_med_min_flase = csv_save_path + "med_min/flase_bayesian_med_min_" + str(epoch) + ".csv"
    save_max_med_flase = csv_save_path + "max_med/flase_bayesian_max_med_" + str(epoch) + ".csv"
    save_med_mean_flase = csv_save_path + "med_mean/flase_bayesian_med_mean_" + str(epoch) + ".csv"
    save_max_max2_flase = csv_save_path + "max_max2/flase_bayesian_max_max2_" + str(epoch) + ".csv"
    save_max_mean_flase = csv_save_path + "max_mean/flase_bayesian_max_mean_" + str(epoch) + ".csv"
    save_mean_min_flase = csv_save_path + "mean_min/flase_bayesian_mean_min_" + str(epoch) + ".csv"

    pd.DataFrame(med_min_list_true).to_csv(save_med_mean_true)
    pd.DataFrame(max_med_list_true).to_csv(save_max_med_true)
    pd.DataFrame(med_min_list_true).to_csv(save_med_min_true)
    pd.DataFrame(max_max2_list_true).to_csv(save_max_max2_true)
    pd.DataFrame(max_mean_list_true).to_csv(save_max_mean_true)
    pd.DataFrame(mean_min_list_true).to_csv(save_mean_min_true)

    pd.DataFrame(med_min_list_flase).to_csv(save_med_mean_flase)
    pd.DataFrame(max_med_list_flase).to_csv(save_max_med_flase)
    pd.DataFrame(med_min_list_flase).to_csv(save_med_min_flase)
    pd.DataFrame(max_max2_list_flase).to_csv(save_max_max2_flase)
    pd.DataFrame(max_mean_list_flase).to_csv(save_max_mean_flase)
    pd.DataFrame(mean_min_list_flase).to_csv(save_mean_min_flase)

    save_std_path_all = csv_save_path + "all/bayesian_std_" + str(epoch) + ".csv"
    save_ent_path_all = csv_save_path + "all/bayesian_ent_" + str(epoch) + ".csv"
    save_kld_path_all = csv_save_path + "all/bayesian_kld_" + str(epoch) + ".csv"

    pd.DataFrame(entropy_true + entropy_flase).to_csv(save_ent_path_all)
    pd.DataFrame(kld_true + kld_flase).to_csv(save_kld_path_all)
    pd.DataFrame(std_true + std_flase).to_csv(save_std_path_all)

    save_max_mean_path_all = csv_save_path + "all/bayesian_max_mean_" + str(epoch) + ".csv"
    save_max_max2_path_all = csv_save_path + "all/bayesian_max_max2_" + str(epoch) + ".csv"
    save_med_mean_path_all = csv_save_path + "all/bayesian_med_mean_" + str(epoch) + ".csv"
    save_med_min_path_all = csv_save_path + "all/bayesian_med_min_" + str(epoch) + ".csv"
    save_max_med_path_all = csv_save_path + "all/bayesian_max_med_" + str(epoch) + ".csv"
    save_mean_min_path_all = csv_save_path + "all/bayesian_mean_min_" + str(epoch) + ".csv"

    pd.DataFrame(max_max2_list_true + max_max2_list_flase).to_csv(save_max_max2_path_all)
    pd.DataFrame(max_med_list_true + max_med_list_flase).to_csv(save_max_med_path_all)
    pd.DataFrame(max_mean_list_true + max_med_list_flase).to_csv(save_max_mean_path_all)
    pd.DataFrame(mean_min_list_true + mean_min_list_flase).to_csv(save_mean_min_path_all)
    pd.DataFrame(med_mean_list_true + med_mean_list_flase).to_csv(save_med_mean_path_all)
    pd.DataFrame(med_min_list_true + med_min_list_flase).to_csv(save_med_min_path_all)

    return entropy_true, entropy_flase, kld_true, kld_flase, std_true, std_flase


def get_mc_dropout_uc(model_directory, ood_data, oos_y, oos_filenames, epoch, architecture_type):
    csv_save_path = "/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/"
    mc = MCDropout(architecture_type=architecture_type, model_directory=model_directory)
    mc.hparam = utils.merge_namedtuples(train_hparam, bayesian_ensemble_hparam)
    pst_pred = mc.predict(ood_data)

    entropy = []
    kld = []
    std = []
    med_min = []
    max_med = []
    med_mean = []
    max_max2 = []
    max_mean = []
    mean_min = []
    pred_prob = []

    for item in pst_pred:
        end_list = []
        for i in range(mc.hparam.n_sampling):
            end_list.append(item[i][0])
        max2 = np.sort(end_list)[-2]
        mean = np.mean(end_list)
        med = np.median(end_list)
        max = np.max(end_list)
        min = np.min(end_list)

        med_mean.append(med - mean)
        max_med.append(max - med)
        max_max2.append(max - max2)
        med_min.append(med - min)
        max_mean.append(max - mean)
        mean_min.append(mean - min)

        pred_prob.append(np.mean(end_list))
        entropy.append(predictive_entropy(end_list))
        kld.append(predictive_kld(end_list, number=mc.hparam.n_sampling)[0])
        std.append(predictive_std(end_list, number=mc.hparam.n_sampling)[0])

    pred_class = []

    entropy_true = []
    entropy_flase = []
    kld_true = []
    kld_flase = []
    std_true = []
    std_flase = []

    med_min_list_flase = []
    max_med_list_flase = []
    med_min_list_true = []
    max_med_list_true = []
    med_mean_list_true = []
    med_mean_list_flase = []
    max_max2_list_true = []
    max_max2_list_flase = []
    max_mean_list_true = []
    max_mean_list_flase = []
    mean_min_list_true = []
    mean_min_list_flase = []

    for i in range(len(oos_filenames)):
        if pred_prob[i] > 0.5:
            pred_class.append(1)
        else:
            pred_class.append(0)

    for i in range(len(oos_filenames)):
        if pred_class[i] == oos_y[i]:
            med_mean_list_true.append([oos_filenames[i], med_mean[i]])
            max_med_list_true.append([oos_filenames[i], max_med[i]])
            max_max2_list_true.append([oos_filenames[i], max_max2[i]])
            med_min_list_true.append([oos_filenames[i], med_min[i]])
            max_mean_list_true.append([oos_filenames[i], max_mean[i]])
            mean_min_list_true.append([oos_filenames[i], mean_min[i]])

            entropy_true.append([oos_filenames[i], entropy[i]])
            kld_true.append([oos_filenames[i], kld[i][0]])
            std_true.append([oos_filenames[i], std[i][0]])
        else:

            med_mean_list_flase.append([oos_filenames[i], med_mean[i]])
            max_med_list_flase.append([oos_filenames[i], max_med[i]])
            max_max2_list_flase.append([oos_filenames[i], max_max2[i]])
            med_min_list_flase.append([oos_filenames[i], med_min[i]])
            max_mean_list_flase.append([oos_filenames[i], max_mean[i]])
            mean_min_list_flase.append([oos_filenames[i], mean_min[i]])

            entropy_flase.append([oos_filenames[i], entropy[i]])
            kld_flase.append([oos_filenames[i], kld[i][0]])
            std_flase.append([oos_filenames[i], std[i][0]])

    save_std_path_true = csv_save_path + "std/true_mcdropout_std_" + str(epoch) + ".csv"
    save_std_path_flase = csv_save_path + "std/flase_mcdropout_std_" + str(epoch) + ".csv"
    save_ent_path_true = csv_save_path + "entropy/true_mcdropout_entropy_" + str(epoch) + ".csv"
    save_ent_path_flase = csv_save_path + "entropy/flase_mcdropout_entropy_" + str(epoch) + ".csv"
    save_kld_path_true = csv_save_path + "kld/true_mcdropout_kld_" + str(epoch) + ".csv"
    save_kld_path_flase = csv_save_path + "kld/flase_mcdropout_kld_" + str(epoch) + ".csv"

    pd.DataFrame(entropy_true).to_csv(save_ent_path_true)
    pd.DataFrame(entropy_flase).to_csv(save_ent_path_flase)
    pd.DataFrame(kld_true).to_csv(save_kld_path_true)
    pd.DataFrame(kld_flase).to_csv(save_kld_path_flase)
    pd.DataFrame(std_true).to_csv(save_std_path_true)
    pd.DataFrame(std_flase).to_csv(save_std_path_flase)

    save_med_min_true = csv_save_path + "med_min/true_mcdropout_med_min_" + str(epoch) + ".csv"
    save_max_med_true = csv_save_path + "max_med/true_mcdropout_max_med_" + str(epoch) + ".csv"
    save_med_mean_true = csv_save_path + "med_mean/true_mcdropout_med_mean_" + str(epoch) + ".csv"
    save_max_max2_true = csv_save_path + "max_max2/true_mcdropout_max_max2_" + str(epoch) + ".csv"
    save_max_mean_true = csv_save_path + "max_mean/true_mcdropout_max_mean_" + str(epoch) + ".csv"
    save_mean_min_true = csv_save_path + "mean_min/true_mcdropout_mean_min_" + str(epoch) + ".csv"
    save_med_min_flase = csv_save_path + "med_min/flase_mcdropout_med_min_" + str(epoch) + ".csv"
    save_max_med_flase = csv_save_path + "max_med/flase_mcdropout_max_med_" + str(epoch) + ".csv"
    save_med_mean_flase = csv_save_path + "med_mean/flase_mcdropout_med_mean_" + str(epoch) + ".csv"
    save_max_max2_flase = csv_save_path + "max_max2/flase_mcdropout_max_max2_" + str(epoch) + ".csv"
    save_max_mean_flase = csv_save_path + "max_mean/flase_mcdropout_max_mean_" + str(epoch) + ".csv"
    save_mean_min_flase = csv_save_path + "mean_min/flase_mcdropout_mean_min_" + str(epoch) + ".csv"

    pd.DataFrame(med_min_list_true).to_csv(save_med_mean_true)
    pd.DataFrame(max_med_list_true).to_csv(save_max_med_true)
    pd.DataFrame(med_min_list_true).to_csv(save_med_min_true)
    pd.DataFrame(max_max2_list_true).to_csv(save_max_max2_true)
    pd.DataFrame(max_mean_list_true).to_csv(save_max_mean_true)
    pd.DataFrame(mean_min_list_true).to_csv(save_mean_min_true)

    pd.DataFrame(med_min_list_flase).to_csv(save_med_mean_flase)
    pd.DataFrame(max_med_list_flase).to_csv(save_max_med_flase)
    pd.DataFrame(med_min_list_flase).to_csv(save_med_min_flase)
    pd.DataFrame(max_max2_list_flase).to_csv(save_max_max2_flase)
    pd.DataFrame(max_mean_list_flase).to_csv(save_max_mean_flase)
    pd.DataFrame(mean_min_list_flase).to_csv(save_mean_min_flase)

    save_std_path_all = csv_save_path + "all/mcdropout_std_" + str(epoch) + ".csv"
    save_ent_path_all = csv_save_path + "all/mcdropout_ent_" + str(epoch) + ".csv"
    save_kld_path_all = csv_save_path + "all/mcdropout_kld_" + str(epoch) + ".csv"
    pd.DataFrame(entropy_true + entropy_flase).to_csv(save_ent_path_all)
    pd.DataFrame(kld_true + kld_flase).to_csv(save_kld_path_all)
    pd.DataFrame(std_true + std_flase).to_csv(save_std_path_all)

    save_max_mean_path_all = csv_save_path + "all/mcdropout_max_mean_" + str(epoch) + ".csv"
    save_max_max2_path_all = csv_save_path + "all/mcdropout_max_max2_" + str(epoch) + ".csv"
    save_med_mean_path_all = csv_save_path + "all/mcdropout_med_mean_" + str(epoch) + ".csv"
    save_med_min_path_all = csv_save_path + "all/mcdropout_med_min_" + str(epoch) + ".csv"
    save_max_med_path_all = csv_save_path + "all/mcdropout_max_med_" + str(epoch) + ".csv"
    save_mean_min_path_all = csv_save_path + "all/mcdropout_mean_min_" + str(epoch) + ".csv"

    pd.DataFrame(max_max2_list_true + max_max2_list_flase).to_csv(save_max_max2_path_all)
    pd.DataFrame(max_med_list_true + max_med_list_flase).to_csv(save_max_med_path_all)
    pd.DataFrame(max_mean_list_true + max_med_list_flase).to_csv(save_max_mean_path_all)
    pd.DataFrame(mean_min_list_true + mean_min_list_flase).to_csv(save_mean_min_path_all)
    pd.DataFrame(med_mean_list_true + med_mean_list_flase).to_csv(save_med_mean_path_all)
    pd.DataFrame(med_min_list_true + med_min_list_flase).to_csv(save_med_min_path_all)

    return entropy_true, entropy_flase, kld_true, kld_flase, std_true, std_flase


def get_deep_ensemble_uc(model_directory, ood_data, oos_y, oos_filenames, epoch, n_members, architecture_type):
    csv_save_path = "/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/"
    d_ensemble = DeepEnsemble(architecture_type=architecture_type, n_members=n_members,
                              model_directory=model_directory)
    d_ensemble_pst_pred = d_ensemble.predict(ood_data)

    entropy = []
    kld = []
    std = []
    pred_prob = []

    med_min = []
    max_med = []
    med_mean = []
    max_max2 = []
    max_mean = []
    mean_min = []

    for item in d_ensemble_pst_pred:
        end_list = []
        for i in range(n_members):
            end_list.append(item[i][0])
        max2 = np.sort(end_list)[-2]
        mean = np.mean(end_list)
        med = np.median(end_list)
        max = np.max(end_list)
        min = np.min(end_list)

        med_mean.append(med - mean)
        max_med.append(max - med)
        max_max2.append(max - max2)
        med_min.append(med - min)
        max_mean.append(max - mean)
        mean_min.append(mean - min)

        pred_prob.append(np.mean(end_list))
        entropy.append(predictive_entropy(end_list))
        kld.append(predictive_kld(end_list, number=n_members)[0])
        std.append(predictive_std(end_list, number=n_members)[0])

    pred_class = []

    entropy_true = []
    entropy_flase = []
    kld_true = []
    kld_flase = []
    std_true = []
    std_flase = []

    med_min_list_flase = []
    max_med_list_flase = []
    med_min_list_true = []
    max_med_list_true = []
    med_mean_list_true = []
    med_mean_list_flase = []
    max_max2_list_true = []
    max_max2_list_flase = []
    max_mean_list_true = []
    max_mean_list_flase = []
    mean_min_list_true = []
    mean_min_list_flase = []

    for i in range(len(oos_filenames)):
        if pred_prob[i] > 0.5:
            pred_class.append(1)
        else:
            pred_class.append(0)

    for i in range(len(oos_filenames)):
        if pred_class[i] == oos_y[i]:
            med_mean_list_true.append([oos_filenames[i], med_mean[i]])
            max_med_list_true.append([oos_filenames[i], max_med[i]])
            max_max2_list_true.append([oos_filenames[i], max_max2[i]])
            med_min_list_true.append([oos_filenames[i], med_min[i]])
            max_mean_list_true.append([oos_filenames[i], max_mean[i]])
            mean_min_list_true.append([oos_filenames[i], mean_min[i]])

            entropy_true.append([oos_filenames[i], entropy[i]])
            kld_true.append([oos_filenames[i], kld[i][0]])
            std_true.append([oos_filenames[i], std[i][0]])
        else:
            med_mean_list_flase.append([oos_filenames[i], med_mean[i]])
            max_med_list_flase.append([oos_filenames[i], max_med[i]])
            max_max2_list_flase.append([oos_filenames[i], max_max2[i]])
            med_min_list_flase.append([oos_filenames[i], med_min[i]])
            max_mean_list_flase.append([oos_filenames[i], max_mean[i]])
            mean_min_list_flase.append([oos_filenames[i], mean_min[i]])

            entropy_flase.append([oos_filenames[i], entropy[i]])
            kld_flase.append([oos_filenames[i], kld[i][0]])
            std_flase.append([oos_filenames[i], std[i][0]])

    save_std_path_true = csv_save_path + "std/true_deepensemble_std_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_std_path_flase = csv_save_path + "std/flase_deepensemble_std_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_ent_path_true = csv_save_path + "entropy/true_deepensemble_entropy_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_ent_path_flase = csv_save_path + "entropy/flase_deepensemble_entropy_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_kld_path_true = csv_save_path + "kld/true_deepensemble_kld_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_kld_path_flase = csv_save_path + "kld/flase_deepensemble_kld_" + str(
        epoch) + "_" + str(n_members) + ".csv"

    pd.DataFrame(entropy_true).to_csv(save_ent_path_true)
    pd.DataFrame(entropy_flase).to_csv(save_ent_path_flase)
    pd.DataFrame(kld_true).to_csv(save_kld_path_true)
    pd.DataFrame(kld_flase).to_csv(save_kld_path_flase)
    pd.DataFrame(std_true).to_csv(save_std_path_true)
    pd.DataFrame(std_flase).to_csv(save_std_path_flase)

    save_med_min_true = csv_save_path + "med_min/true_deepensemble_med_min_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_med_true = csv_save_path + "max_med/true_deepensemble_max_med_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_med_mean_true = csv_save_path + "med_mean/true_deepensemble_med_mean_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_max2_true = csv_save_path + "max_max2/true_deepensemble_max_max2_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_mean_true = csv_save_path + "max_mean/true_deepensemble_max_mean_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_mean_min_true = csv_save_path + "mean_min/true_deepensemble_mean_min_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_med_min_flase = csv_save_path + "med_min/flase_deepensemble_med_min_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_med_flase = csv_save_path + "max_med/flase_deepensemble_max_med_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_med_mean_flase = csv_save_path + "med_mean/flase_deepensemble_med_mean_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_max2_flase = csv_save_path + "max_max2/flase_deepensemble_max_max2_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_mean_flase = csv_save_path + "max_mean/flase_deepensemble_max_mean_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_mean_min_flase = csv_save_path + "mean_min/flase_deepensemble_mean_min_" + str(
        epoch) + "_" + str(n_members) + ".csv"

    pd.DataFrame(med_min_list_true).to_csv(save_med_mean_true)
    pd.DataFrame(max_med_list_true).to_csv(save_max_med_true)
    pd.DataFrame(med_min_list_true).to_csv(save_med_min_true)
    pd.DataFrame(max_max2_list_true).to_csv(save_max_max2_true)
    pd.DataFrame(max_mean_list_true).to_csv(save_max_mean_true)
    pd.DataFrame(mean_min_list_true).to_csv(save_mean_min_true)

    pd.DataFrame(med_min_list_flase).to_csv(save_med_mean_flase)
    pd.DataFrame(max_med_list_flase).to_csv(save_max_med_flase)
    pd.DataFrame(med_min_list_flase).to_csv(save_med_min_flase)
    pd.DataFrame(max_max2_list_flase).to_csv(save_max_max2_flase)
    pd.DataFrame(max_mean_list_flase).to_csv(save_max_mean_flase)
    pd.DataFrame(mean_min_list_flase).to_csv(save_mean_min_flase)

    save_std_path_all = csv_save_path + "all/deepensemble_std_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_ent_path_all = csv_save_path + "all/deepensemble_ent_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_kld_path_all = csv_save_path + "all/deepensemble_kld_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    pd.DataFrame(entropy_true + entropy_flase).to_csv(save_ent_path_all)
    pd.DataFrame(kld_true + kld_flase).to_csv(save_kld_path_all)
    pd.DataFrame(std_true + std_flase).to_csv(save_std_path_all)

    save_max_mean_path_all = csv_save_path + "all/deepensemble_max_mean_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_max2_path_all = csv_save_path + "all/deepensemble_max_max2_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_med_mean_path_all = csv_save_path + "all/deepensemble_med_mean_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_med_min_path_all = csv_save_path + "all/deepensemble_med_min_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_med_path_all = csv_save_path + "all/deepensemble_max_med_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_mean_min_path_all = csv_save_path + "all/deepensemble_mean_min_" + str(
        epoch) + "_" + str(n_members) + ".csv"

    pd.DataFrame(max_max2_list_true + max_max2_list_flase).to_csv(save_max_max2_path_all)
    pd.DataFrame(max_med_list_true + max_med_list_flase).to_csv(save_max_med_path_all)
    pd.DataFrame(max_mean_list_true + max_med_list_flase).to_csv(save_max_mean_path_all)
    pd.DataFrame(mean_min_list_true + mean_min_list_flase).to_csv(save_mean_min_path_all)
    pd.DataFrame(med_mean_list_true + med_mean_list_flase).to_csv(save_med_mean_path_all)
    pd.DataFrame(med_min_list_true + med_min_list_flase).to_csv(save_med_min_path_all)

    return entropy_true, entropy_flase, kld_true, kld_flase, std_true, std_flase


def get_weight_ensemble_uc(model_directory, ood_data, oos_y, oos_filenames, epoch, n_members, architecture_type):
    csv_save_path = "/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/"
    w_ensemble = WeightedDeepEnsemble(architecture_type=architecture_type, n_members=n_members,
                                      model_directory=model_directory)
    w_ensemble_pst_pred = w_ensemble.predict(ood_data)

    entropy = []
    kld = []
    std = []
    pred_prob = []

    med_min = []
    max_med = []
    med_mean = []
    max_max2 = []
    max_mean = []
    mean_min = []

    for item in w_ensemble_pst_pred[0]:
        end_list = []
        for i in range(n_members):
            end_list.append(item[i][0])

        max2 = np.sort(end_list)[-2]
        mean = np.mean(end_list)
        med = np.median(end_list)
        max = np.max(end_list)
        min = np.min(end_list)

        med_mean.append(med - mean)
        max_med.append(max - med)
        max_max2.append(max - max2)
        med_min.append(med - min)
        max_mean.append(max - mean)
        mean_min.append(mean - min)

        pred_prob.append(np.mean(end_list))
        entropy.append(predictive_entropy(end_list))
        kld.append(predictive_kld(end_list, number=n_members)[0])
        std.append(predictive_std(end_list, number=n_members)[0])

    pred_class = []

    entropy_true = []
    entropy_flase = []
    kld_true = []
    kld_flase = []
    std_true = []
    std_flase = []

    med_min_list_flase = []
    max_med_list_flase = []
    med_min_list_true = []
    max_med_list_true = []
    med_mean_list_true = []
    med_mean_list_flase = []
    max_max2_list_true = []
    max_max2_list_flase = []
    max_mean_list_true = []
    max_mean_list_flase = []
    mean_min_list_true = []
    mean_min_list_flase = []

    for i in range(len(oos_filenames)):
        if pred_prob[i] > 0.5:
            pred_class.append(1)
        else:
            pred_class.append(0)

    for i in range(len(oos_filenames)):
        if pred_class[i] == oos_y[i]:

            med_mean_list_true.append([oos_filenames[i], med_mean[i]])
            max_med_list_true.append([oos_filenames[i], max_med[i]])
            max_max2_list_true.append([oos_filenames[i], max_max2[i]])
            med_min_list_true.append([oos_filenames[i], med_min[i]])
            max_mean_list_true.append([oos_filenames[i], max_mean[i]])
            mean_min_list_true.append([oos_filenames[i], mean_min[i]])

            entropy_true.append([oos_filenames[i], entropy[i]])
            kld_true.append([oos_filenames[i], kld[i][0]])
            std_true.append([oos_filenames[i], std[i][0]])
        else:

            med_mean_list_flase.append([oos_filenames[i], med_mean[i]])
            max_med_list_flase.append([oos_filenames[i], max_med[i]])
            max_max2_list_flase.append([oos_filenames[i], max_max2[i]])
            med_min_list_flase.append([oos_filenames[i], med_min[i]])
            max_mean_list_flase.append([oos_filenames[i], max_mean[i]])
            mean_min_list_flase.append([oos_filenames[i], mean_min[i]])

            entropy_flase.append([oos_filenames[i], entropy[i]])
            kld_flase.append([oos_filenames[i], kld[i][0]])
            std_flase.append([oos_filenames[i], std[i][0]])

    save_std_path_true = csv_save_path + "std/true_weightensemble_std_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_std_path_flase = csv_save_path + "std/flase_weightensemble_std_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_ent_path_true = csv_save_path + "entropy/true_weightensemble_entropy_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_ent_path_flase = csv_save_path + "entropy/flase_weightensemble_entropy_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_kld_path_true = csv_save_path + "kld/true_weightensemble_kld_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_kld_path_flase = csv_save_path + "kld/flase_weightensemble_kld_" + str(
        epoch) + "_" + str(n_members) + ".csv"

    pd.DataFrame(entropy_true).to_csv(save_ent_path_true)
    pd.DataFrame(entropy_flase).to_csv(save_ent_path_flase)
    pd.DataFrame(kld_true).to_csv(save_kld_path_true)
    pd.DataFrame(kld_flase).to_csv(save_kld_path_flase)
    pd.DataFrame(std_true).to_csv(save_std_path_true)
    pd.DataFrame(std_flase).to_csv(save_std_path_flase)

    save_med_min_true = csv_save_path + "med_min/true_weightensemble_med_min_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_med_true = csv_save_path + "max_med/true_weightensemble_max_med_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_med_mean_true = csv_save_path + "med_mean/true_weightensemble_med_mean_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_max2_true = csv_save_path + "max_max2/true_weightensemble_max_max2_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_mean_true = csv_save_path + "max_mean/true_weightensemble_max_mean_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_mean_min_true = csv_save_path + "mean_min/true_weightensemble_mean_min_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_med_min_flase = csv_save_path + "med_min/flase_weightensemble_med_min_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_med_flase = csv_save_path + "max_med/flase_weightensemble_max_med_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_med_mean_flase = csv_save_path + "med_mean/flase_weightensemble_med_mean_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_max2_flase = csv_save_path + "max_max2/flase_weightensemble_max_max2_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_mean_flase = csv_save_path + "max_mean/flase_weightensemble_max_mean_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_mean_min_flase = csv_save_path + "mean_min/flase_weightensemble_mean_min_" + str(
        epoch) + "_" + str(n_members) + ".csv"

    pd.DataFrame(med_min_list_true).to_csv(save_med_mean_true)
    pd.DataFrame(max_med_list_true).to_csv(save_max_med_true)
    pd.DataFrame(med_min_list_true).to_csv(save_med_min_true)
    pd.DataFrame(max_max2_list_true).to_csv(save_max_max2_true)
    pd.DataFrame(max_mean_list_true).to_csv(save_max_mean_true)
    pd.DataFrame(mean_min_list_true).to_csv(save_mean_min_true)

    pd.DataFrame(med_min_list_flase).to_csv(save_med_mean_flase)
    pd.DataFrame(max_med_list_flase).to_csv(save_max_med_flase)
    pd.DataFrame(med_min_list_flase).to_csv(save_med_min_flase)
    pd.DataFrame(max_max2_list_flase).to_csv(save_max_max2_flase)
    pd.DataFrame(max_mean_list_flase).to_csv(save_max_mean_flase)
    pd.DataFrame(mean_min_list_flase).to_csv(save_mean_min_flase)

    save_std_path_all = csv_save_path + "all/weightensemble_std_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_ent_path_all = csv_save_path + "all/weightensemble_ent_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_kld_path_all = csv_save_path + "all/weightensemble_kld_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    pd.DataFrame(entropy_true + entropy_flase).to_csv(save_ent_path_all)
    pd.DataFrame(kld_true + kld_flase).to_csv(save_kld_path_all)
    pd.DataFrame(std_true + std_flase).to_csv(save_std_path_all)

    save_max_mean_path_all = csv_save_path + "all/weightensemble_max_mean_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_max2_path_all = csv_save_path + "all/weightensemble_max_max2_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_med_mean_path_all = csv_save_path + "all/weightensemble_med_mean_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_med_min_path_all = csv_save_path + "all/weightensemble_med_min_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_max_med_path_all = csv_save_path + "all/weightensemble_max_med_" + str(
        epoch) + "_" + str(n_members) + ".csv"
    save_mean_min_path_all = csv_save_path + "all/weightensemble_mean_min_" + str(
        epoch) + "_" + str(n_members) + ".csv"

    pd.DataFrame(max_max2_list_true + max_max2_list_flase).to_csv(save_max_max2_path_all)
    pd.DataFrame(max_med_list_true + max_med_list_flase).to_csv(save_max_med_path_all)
    pd.DataFrame(max_mean_list_true + max_med_list_flase).to_csv(save_max_mean_path_all)
    pd.DataFrame(mean_min_list_true + mean_min_list_flase).to_csv(save_mean_min_path_all)
    pd.DataFrame(med_mean_list_true + med_mean_list_flase).to_csv(save_med_mean_path_all)
    pd.DataFrame(med_min_list_true + med_min_list_flase).to_csv(save_med_min_path_all)

    return entropy_true, entropy_flase, kld_true, kld_flase, std_true, std_flase
