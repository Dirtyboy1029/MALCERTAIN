# -*- coding: utf-8 -*- 
# @Time : 2022/9/23 9:02 
# @Author : DirtyBoy 
# @File : utils_new.py
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


def get_label(model_path, prist_data, adv_data, prist_filenames, adv_filenames, architecture_type):
    base_model = Vanilla(architecture_type=architecture_type, model_directory=model_path)
    pst_pred_prob = base_model.predict(prist_data, use_prob=True)
    adv_pred_prob = base_model.predict(adv_data, use_prob=True)
    pst_pred_class = []
    adv_pred_class = []
    pred_true_name = []
    pred_flase_name = []
    pred_true_y = []
    pred_flase_y = []
    for i in range(len(prist_filenames)):
        if pst_pred_prob[i] > 0.5:
            pst_pred_class.append(1)
        else:
            pst_pred_class.append(0)
    for i in range(len(adv_filenames)):
        if adv_pred_prob[i] > 0.5:
            adv_pred_class.append(1)
        else:
            adv_pred_class.append(0)

    for i in range(len(prist_filenames)):
        if pst_pred_class[i] == 1:
            pred_true_name.append(prist_filenames[i])
            pred_true_y.append(pst_pred_class[i])
        else:
            pred_flase_name.append(prist_filenames[i])
            pred_flase_y.append(pst_pred_class[i])
    for i in range(len(adv_filenames)):
        if adv_pred_class[i] == 1:
            pred_true_name.append(adv_filenames[i])
            pred_true_y.append(adv_pred_class[i])
        else:
            pred_flase_name.append(adv_filenames[i])
            pred_flase_y.append(adv_pred_class[i])

    print("prist sample true:", np.sum(pst_pred_class) / len(prist_filenames))
    print("adv sample true:", np.sum(adv_pred_class) / len(adv_filenames))
    print("所有样本预测正确：", len(pred_true_name))
    print("所有样本预测错误：", len(pred_flase_name))
    df_true = pd.DataFrame()
    df_true["apk_name"] = pred_true_name
    df_true["label"] = [1.0] * len(pred_true_name)
    df_true["really_label"] = [1.0] * len(pred_true_name)
    df_true["pred_label"] = np.array(pred_true_y)
    df_flase = pd.DataFrame()
    df_flase["apk_name"] = pred_flase_name
    df_flase["label"] = [0.0] * len(pred_flase_name)
    df_flase["really_label"] = [1.0] * len(pred_flase_name)
    df_flase["pred_label"] = np.array(pred_flase_y)
    df_true.to_csv("/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/label/pred_true.csv")
    df_flase.to_csv(
        "/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/label/pred_flase.csv")
    return pred_true_name, pred_flase_name


def epoch_ensemble_uc(base_path, prist_data, adv_data, prist_filenames, adv_filenames, architecture_type,
                      calibration=False,
                      true_flase=False):
    temperature_list = [2.9304572343836255, 3.2258611917495528, 3.1972519159318122, 3.491626143454783,
                        3.081241726874375]
    txt_save_path = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/"
    csv_save_path = "/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/"
    model_name_list = os.listdir(base_path)
    model_path_list = [base_path + item for item in model_name_list]
    model_list = [Vanilla(architecture_type=architecture_type,
                          model_directory=item) for item in model_path_list]

    if calibration:
        pst_pred = [apply_temperature_scaling(temperature_list[i], model_list[i].predict(prist_data, use_prob=True)) for
                    i in range(len(model_list))]
        adv_pred = [apply_temperature_scaling(temperature_list[i], model_list[i].predict(adv_data, use_prob=True)) for i
                    in range(len(model_list))]

        save_std_path_pst = txt_save_path + "c_std/epoch_ensemble_std_pst_c.txt"
        save_std_path_adv = txt_save_path + "c_std/epoch_ensemble_std_adv_c.txt"
        save_ent_path_pst = txt_save_path + "c_entropy/epoch_ensemble_pst_c.txt"
        save_ent_path_adv = txt_save_path + "c_entropy/epoch_ensemble_adv_c.txt"
        save_kld_path_pst = txt_save_path + "c_kld/epoch_ensemble_pst_c.txt"
        save_kld_path_adv = txt_save_path + "c_kld/epoch_ensemble_adv_c.txt"

        save_std_path_true = csv_save_path + "c_std/epoch_ensemble_std_true_c.csv"
        save_std_path_flase = csv_save_path + "c_std/epoch_ensemble_std_flase_c.csv"
        save_ent_path_true = csv_save_path + "c_entropy/epoch_ensemble_ent_true_c.csv"
        save_ent_path_flase = csv_save_path + "c_entropy/epoch_ensemble_ent_flase_c.csv"
        save_kld_path_true = csv_save_path + "c_kld/epoch_ensemble_kld_true_c.csv"
        save_kld_path_flase = csv_save_path + "c_kld/epoch_ensemble_kld_flase_c.csv"

    else:
        pst_pred = [item.predict(prist_data, use_prob=True) for item in model_list]
        adv_pred = [item.predict(adv_data, use_prob=True) for item in model_list]

        save_std_path_pst = txt_save_path + "std/epoch_ensemble_std_pst.txt"
        save_std_path_adv = txt_save_path + "std/epoch_ensemble_std_adv.txt"
        save_ent_path_pst = txt_save_path + "entropy/epoch_ensemble_pst.txt"
        save_ent_path_adv = txt_save_path + "entropy/epoch_ensemble_adv.txt"
        save_kld_path_pst = txt_save_path + "kld/epoch_ensemble_pst.txt"
        save_kld_path_adv = txt_save_path + "kld/epoch_ensemble_adv.txt"

        save_std_path_true = csv_save_path + "std/epoch_ensemble_std_true.csv"
        save_std_path_flase = csv_save_path + "std/epoch_ensemble_std_flase.csv"
        save_ent_path_true = csv_save_path + "entropy/epoch_ensemble_ent_true.csv"
        save_ent_path_flase = csv_save_path + "entropy/epoch_ensemble_ent_flase.csv"
        save_kld_path_true = csv_save_path + "kld/epoch_ensemble_kld_true.csv"
        save_kld_path_flase = csv_save_path + "kld/epoch_ensemble_kld_flase.csv"

    entropy_adv = []
    entropy_pst = []
    kld_adv = []
    kld_pst = []
    std_adv = []
    std_pst = []
    pst_pred_prob = []
    adv_pred_prob = []

    for i in range(len(prist_filenames)):
        tmp = []
        for item in pst_pred:
            tmp.append(item[i][0])
        pst_pred_prob.append(np.mean(tmp))
        std_pst.append(predictive_std(tmp, number=len(model_list))[0])
        entropy_pst.append(predictive_entropy(tmp))
        kld_pst.append(predictive_kld(tmp, number=len(model_list))[0])
    for i in range(len(adv_filenames)):
        tmp = []
        for item in adv_pred:
            tmp.append(item[i][0])
        adv_pred_prob.append(np.mean(tmp))
        std_adv.append(predictive_std(tmp, number=len(model_list))[0])
        entropy_adv.append(predictive_entropy(tmp))
        kld_adv.append(predictive_kld(tmp, number=len(model_list))[0])

    if true_flase:
        pst_pred_class = []
        adv_pred_class = []
        entropy_true = []
        entropy_flase = []
        kld_true = []
        kld_flase = []
        std_true = []
        std_flase = []
        for i in range(len(adv_filenames)):
            if pst_pred_prob[i] > 0.5:
                pst_pred_class.append(1)
            else:
                pst_pred_class.append(0)
            if adv_pred_prob[i] > 0.5:
                adv_pred_class.append(1)
            else:
                adv_pred_class.append(0)

        for i in range(len(prist_filenames)):
            if pst_pred_class[i] == 1:
                entropy_true.append([prist_filenames[i], entropy_pst[i]])
                kld_true.append([prist_filenames[i], kld_pst[i][0]])
                std_true.append([prist_filenames[i], std_pst[i][0]])
            else:
                entropy_flase.append([prist_filenames[i], entropy_pst[i]])
                kld_flase.append([prist_filenames[i], kld_pst[i][0]])
                std_flase.append([prist_filenames[i], std_pst[i][0]])
            if adv_pred_class[i] == 1:
                entropy_true.append([adv_filenames[i], entropy_adv[i]])
                kld_true.append([adv_filenames[i], kld_adv[i][0]])
                std_true.append([adv_filenames[i], std_adv[i][0]])
            else:
                entropy_flase.append([adv_filenames[i], entropy_adv[i]])
                kld_flase.append([adv_filenames[i], kld_adv[i][0]])
                std_flase.append([adv_filenames[i], std_adv[i][0]])

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
    else:
        np.savetxt(save_kld_path_pst, kld_pst)
        np.savetxt(save_kld_path_adv, kld_adv)
        np.savetxt(save_ent_path_pst, entropy_pst)
        np.savetxt(save_ent_path_adv, entropy_adv)
        np.savetxt(save_std_path_pst, std_pst)
        np.savetxt(save_std_path_adv, std_adv)
        return entropy_pst, entropy_adv, kld_pst, kld_adv, std_pst, std_adv


def get_bay_uc(prist_data, adv_data, model_directory, epoch, prist_filenames, adv_filenames, true_flase=False,architecture_type="dnn"):
    csv_save_path = "/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/"
    bay = BayesianEnsemble(architecture_type=architecture_type, model_directory=model_directory)
    bay.hparam = utils.merge_namedtuples(train_hparam, bayesian_ensemble_hparam)
    bay_pst_pred = bay.predict(prist_data)
    bay_adv_pred = bay.predict(adv_data)

    entropy_list_adv = []
    entropy_list_pst = []
    kld_list_adv = []
    kld_list_pst = []
    std_list_adv = []
    std_list_pst = []
    pst_pred_prob = []
    adv_pred_prob = []

    med_min_list_adv = []
    max_med_list_adv = []
    med_min_list_pst = []
    max_med_list_pst = []
    med_mean_list_pst = []
    med_mean_list_adv = []
    max_max2_list_pst = []
    max_max2_list_adv = []
    max_mean_list_pst = []
    max_mean_list_adv = []
    mean_min_list_pst = []
    mean_min_list_adv = []

    for item in bay_pst_pred:
        end_list = []
        for i in range(bay.hparam.n_sampling):
            end_list.append(item[i][0])
        max2 = np.sort(end_list)[-2]
        mean = np.mean(end_list)
        med = np.median(end_list)
        max = np.max(end_list)
        min = np.min(end_list)

        med_mean_list_pst.append(med - mean)
        max_med_list_pst.append(max - med)
        max_max2_list_pst.append(max - max2)
        med_min_list_pst.append(med - min)
        max_mean_list_pst.append(max - mean)
        mean_min_list_pst.append(mean - min)

        pst_pred_prob.append(mean)
        entropy_list_pst.append(predictive_entropy(end_list))
        kld_list_pst.append(predictive_kld(end_list, number=bay.hparam.n_sampling)[0])
        std_list_pst.append(predictive_std(end_list, number=bay.hparam.n_sampling)[0])
    for item in bay_adv_pred:
        end_list = []
        for i in range(bay.hparam.n_sampling):
            end_list.append(item[i][0])

        max2 = np.sort(end_list)[-2]
        mean = np.mean(end_list)
        med = np.median(end_list)
        max = np.max(end_list)
        min = np.min(end_list)

        med_mean_list_adv.append(med - mean)
        max_med_list_adv.append(max - med)
        max_max2_list_adv.append(max - max2)
        med_min_list_adv.append(med - min)
        max_mean_list_adv.append(max - mean)
        mean_min_list_adv.append(mean - min)

        adv_pred_prob.append(np.mean(end_list))
        entropy_list_adv.append(predictive_entropy(end_list))
        kld_list_adv.append(predictive_kld(end_list, number=bay.hparam.n_sampling)[0])
        std_list_adv.append(predictive_std(end_list, number=bay.hparam.n_sampling)[0])

    if true_flase:
        pst_pred_class = []
        adv_pred_class = []
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

        for i in range(len(adv_filenames)):
            if pst_pred_prob[i] > 0.5:
                pst_pred_class.append(1)
            else:
                pst_pred_class.append(0)
            if adv_pred_prob[i] > 0.5:
                adv_pred_class.append(1)
            else:
                adv_pred_class.append(0)

        for i in range(len(prist_filenames)):
            if pst_pred_class[i] == 1:

                med_mean_list_true.append([prist_filenames[i], med_mean_list_pst[i]])
                max_med_list_true.append([prist_filenames[i], max_med_list_pst[i]])
                max_max2_list_true.append([prist_filenames[i], max_max2_list_pst[i]])
                med_min_list_true.append([prist_filenames[i], med_min_list_pst[i]])
                max_mean_list_true.append([prist_filenames[i], max_mean_list_pst[i]])
                mean_min_list_true.append([prist_filenames[i], mean_min_list_pst[i]])

                entropy_true.append([prist_filenames[i], entropy_list_pst[i]])
                kld_true.append([prist_filenames[i], kld_list_pst[i][0]])
                std_true.append([prist_filenames[i], std_list_pst[i][0]])
            else:
                med_mean_list_flase.append([prist_filenames[i], med_mean_list_pst[i]])
                max_med_list_flase.append([prist_filenames[i], max_med_list_pst[i]])
                max_max2_list_flase.append([prist_filenames[i], max_max2_list_pst[i]])
                med_min_list_flase.append([prist_filenames[i], med_min_list_pst[i]])
                max_mean_list_flase.append([prist_filenames[i], max_mean_list_pst[i]])
                mean_min_list_flase.append([prist_filenames[i], mean_min_list_pst[i]])

                entropy_flase.append([prist_filenames[i], entropy_list_pst[i]])
                kld_flase.append([prist_filenames[i], kld_list_pst[i][0]])
                std_flase.append([prist_filenames[i], std_list_pst[i][0]])
            if adv_pred_class[i] == 1:
                med_mean_list_true.append([adv_filenames[i], med_mean_list_adv[i]])
                max_med_list_true.append([adv_filenames[i], max_med_list_adv[i]])
                max_max2_list_true.append([adv_filenames[i], max_max2_list_adv[i]])
                med_min_list_true.append([adv_filenames[i], med_min_list_adv[i]])
                max_mean_list_true.append([adv_filenames[i], max_mean_list_adv[i]])
                mean_min_list_true.append([adv_filenames[i], mean_min_list_adv[i]])

                entropy_true.append([adv_filenames[i], entropy_list_adv[i]])
                kld_true.append([adv_filenames[i], kld_list_adv[i][0]])
                std_true.append([adv_filenames[i], std_list_adv[i][0]])
            else:
                med_mean_list_flase.append([adv_filenames[i], med_mean_list_adv[i]])
                max_med_list_flase.append([adv_filenames[i], max_med_list_adv[i]])
                max_max2_list_flase.append([adv_filenames[i], max_max2_list_adv[i]])
                med_min_list_flase.append([adv_filenames[i], med_min_list_adv[i]])
                max_mean_list_flase.append([adv_filenames[i], max_mean_list_adv[i]])
                mean_min_list_flase.append([adv_filenames[i], mean_min_list_adv[i]])

                entropy_flase.append([adv_filenames[i], entropy_list_adv[i]])
                kld_flase.append([adv_filenames[i], kld_list_adv[i][0]])
                std_flase.append([adv_filenames[i], std_list_adv[i][0]])

        save_std_path_true = csv_save_path + "std/true_bayesian_std_" + epoch + ".csv"
        save_std_path_flase = csv_save_path + "std/flase_bayesian_std_" + epoch + ".csv"
        save_ent_path_true = csv_save_path + "entropy/true_bayesian_entropy_" + epoch + ".csv"
        save_ent_path_flase = csv_save_path + "entropy/flase_bayesian_entropy_" + epoch + ".csv"
        save_kld_path_true = csv_save_path + "kld/true_bayesian_kld_" + epoch + ".csv"
        save_kld_path_flase = csv_save_path + "kld/flase_bayesian_kld_" + epoch + ".csv"

        pd.DataFrame(entropy_true).to_csv(save_ent_path_true)
        pd.DataFrame(entropy_flase).to_csv(save_ent_path_flase)
        pd.DataFrame(kld_true).to_csv(save_kld_path_true)
        pd.DataFrame(kld_flase).to_csv(save_kld_path_flase)
        pd.DataFrame(std_true).to_csv(save_std_path_true)
        pd.DataFrame(std_flase).to_csv(save_std_path_flase)

        save_med_min_true = csv_save_path + "med_min/true_bayesian_med_min_" + epoch + ".csv"
        save_max_med_true = csv_save_path + "max_med/true_bayesian_max_med_" + epoch + ".csv"
        save_med_mean_true = csv_save_path + "med_mean/true_bayesian_med_mean_" + epoch + ".csv"
        save_max_max2_true = csv_save_path + "max_max2/true_bayesian_max_max2_" + epoch + ".csv"
        save_max_mean_true = csv_save_path + "max_mean/true_bayesian_max_mean_" + epoch + ".csv"
        save_mean_min_true = csv_save_path + "mean_min/true_bayesian_mean_min_" + epoch + ".csv"
        save_med_min_flase = csv_save_path + "med_min/flase_bayesian_med_min_" + epoch + ".csv"
        save_max_med_flase = csv_save_path + "max_med/flase_bayesian_max_med_" + epoch + ".csv"
        save_med_mean_flase = csv_save_path + "med_mean/flase_bayesian_med_mean_" + epoch + ".csv"
        save_max_max2_flase = csv_save_path + "max_max2/flase_bayesian_max_max2_" + epoch + ".csv"
        save_max_mean_flase = csv_save_path + "max_mean/flase_bayesian_max_mean_" + epoch + ".csv"
        save_mean_min_flase = csv_save_path + "mean_min/flase_bayesian_mean_min_" + epoch + ".csv"

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

        save_std_path_all = csv_save_path + "all/bayesian_std_" + epoch + ".csv"
        save_ent_path_all = csv_save_path + "all/bayesian_ent_" + epoch + ".csv"
        save_kld_path_all = csv_save_path + "all/bayesian_kld_" + epoch + ".csv"

        pd.DataFrame(entropy_true + entropy_flase).to_csv(save_ent_path_all)
        pd.DataFrame(kld_true + kld_flase).to_csv(save_kld_path_all)
        pd.DataFrame(std_true + std_flase).to_csv(save_std_path_all)

        save_max_mean_path_all = csv_save_path + "all/bayesian_max_mean_" + epoch + ".csv"
        save_max_max2_path_all = csv_save_path + "all/bayesian_max_max2_" + epoch + ".csv"
        save_med_mean_path_all = csv_save_path + "all/bayesian_med_mean_" + epoch + ".csv"
        save_med_min_path_all = csv_save_path + "all/bayesian_med_min_" + epoch + ".csv"
        save_max_med_path_all = csv_save_path + "all/bayesian_max_med_" + epoch + ".csv"
        save_mean_min_path_all = csv_save_path + "all/bayesian_mean_min_" + epoch + ".csv"

        pd.DataFrame(max_max2_list_true + max_max2_list_flase).to_csv(save_max_max2_path_all)
        pd.DataFrame(max_med_list_true + max_med_list_flase).to_csv(save_max_med_path_all)
        pd.DataFrame(max_mean_list_true + max_med_list_flase).to_csv(save_max_mean_path_all)
        pd.DataFrame(mean_min_list_true + mean_min_list_flase).to_csv(save_mean_min_path_all)
        pd.DataFrame(med_mean_list_true + med_mean_list_flase).to_csv(save_med_mean_path_all)
        pd.DataFrame(med_min_list_true + med_min_list_flase).to_csv(save_med_min_path_all)

        return entropy_true, entropy_flase, kld_true, kld_flase, std_true, std_flase
    else:
        pst_kld_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/kld/pst_bayesian_kld_" + epoch + ".txt"
        adv_kld_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/kld/adv_bayesian_kld_" + epoch + ".txt"
        pst_std_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/std/pst_bayesian_std_" + epoch + ".txt"
        adv_std_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/std/adv_bayesian_std_" + epoch + ".txt"
        pst_ent_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/entropy/pst_bayesian_entropy_" + epoch + ".txt"
        adv_ent_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/entropy/adv_bayesian_entropy_" + epoch + ".txt"
        np.savetxt(pst_std_file_name, std_list_pst)
        np.savetxt(adv_std_file_name, std_list_adv)
        np.savetxt(pst_kld_file_name, kld_list_pst)
        np.savetxt(adv_kld_file_name, kld_list_adv)
        np.savetxt(pst_ent_file_name, entropy_list_pst)
        np.savetxt(adv_ent_file_name, entropy_list_adv)

        adv_med_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_min/adv_bayesian_med_min_" + epoch + ".txt"
        pst_med_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_min/pst_bayesian_med_min_" + epoch + ".txt"
        adv_max_med_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_med/adv_bayesian_max_med_" + epoch + ".txt"
        pst_max_med_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_med/pst_bayesian_max_med_" + epoch + ".txt"
        adv_med_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_mean/adv_bayesian_med_mean_" + epoch + ".txt"
        pst_med_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_mean/pst_bayesian_med_mean_" + epoch + ".txt"
        adv_max_max2_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_max2/adv_bayesian_max_max2_" + epoch + ".txt"
        pst_max_max2_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_max2/pst_bayesian_max_max2_" + epoch + ".txt"
        adv_max_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_mean/adv_bayesian_max_mean_" + epoch + ".txt"
        pst_max_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_mean/pst_bayesian_max_mean_" + epoch + ".txt"
        adv_mean_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/mean_min/adv_bayesian_mean_min_" + epoch + ".txt"
        pst_mean_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/mean_min/pst_bayesian_mean_min_" + epoch + ".txt"

        np.savetxt(adv_med_min_file_name, med_min_list_adv)
        np.savetxt(pst_med_min_file_name, med_min_list_pst)
        np.savetxt(adv_max_med_file_name, max_med_list_adv)
        np.savetxt(pst_max_med_file_name, max_med_list_pst)
        np.savetxt(adv_med_mean_file_name, med_mean_list_adv)
        np.savetxt(pst_med_mean_file_name, med_mean_list_pst)
        np.savetxt(adv_max_max2_file_name, max_max2_list_adv)
        np.savetxt(pst_max_max2_file_name, max_max2_list_pst)
        np.savetxt(adv_max_mean_file_name, max_mean_list_adv)
        np.savetxt(pst_max_mean_file_name, max_mean_list_pst)
        np.savetxt(adv_mean_min_file_name, mean_min_list_adv)
        np.savetxt(pst_mean_min_file_name, mean_min_list_pst)
        return entropy_list_pst, entropy_list_adv, kld_list_pst, kld_list_adv, std_list_pst, std_list_adv, \
               med_min_list_adv, max_med_list_adv, med_min_list_pst, max_med_list_pst, med_mean_list_pst, \
               med_mean_list_adv, max_max2_list_pst, max_max2_list_adv, max_mean_list_pst, max_mean_list_adv, \
               mean_min_list_pst, mean_min_list_adv


def get_mc_dropout_uc(prist_data, adv_data, model_directory, epoch, prist_filenames, adv_filenames, architecture_type="dnn",
                      true_flase=False):
    csv_save_path = "/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/"
    mc = MCDropout(architecture_type=architecture_type, model_directory=model_directory)
    mc.hparam = utils.merge_namedtuples(train_hparam, bayesian_ensemble_hparam)
    mc_pst_pred = mc.predict(prist_data)
    mc_adv_pred = mc.predict(adv_data)

    entropy_list_adv = []
    entropy_list_pst = []
    kld_list_adv = []
    kld_list_pst = []
    std_list_adv = []
    std_list_pst = []
    pst_pred_prob = []
    adv_pred_prob = []

    med_min_list_adv = []
    max_med_list_adv = []
    med_min_list_pst = []
    max_med_list_pst = []
    med_mean_list_pst = []
    med_mean_list_adv = []
    max_max2_list_pst = []
    max_max2_list_adv = []
    max_mean_list_pst = []
    max_mean_list_adv = []
    mean_min_list_pst = []
    mean_min_list_adv = []

    for item in mc_pst_pred:
        end_list = []
        for i in range(mc.hparam.n_sampling):
            end_list.append(item[i][0])
        max2 = np.sort(end_list)[-2]
        mean = np.mean(end_list)
        med = np.median(end_list)
        max = np.max(end_list)
        min = np.min(end_list)

        med_mean_list_pst.append(med - mean)
        max_med_list_pst.append(max - med)
        max_max2_list_pst.append(max - max2)
        med_min_list_pst.append(med - min)
        max_mean_list_pst.append(max - mean)
        mean_min_list_pst.append(mean - min)

        pst_pred_prob.append(np.mean(end_list))
        entropy_list_pst.append(predictive_entropy(end_list))
        kld_list_pst.append(predictive_kld(end_list, number=mc.hparam.n_sampling)[0])
        std_list_pst.append(predictive_std(end_list, number=mc.hparam.n_sampling)[0])
    for item in mc_adv_pred:
        end_list = []
        for i in range(mc.hparam.n_sampling):
            end_list.append(item[i][0])

        max2 = np.sort(end_list)[-2]
        mean = np.mean(end_list)
        med = np.median(end_list)
        max = np.max(end_list)
        min = np.min(end_list)

        med_mean_list_adv.append(med - mean)
        max_med_list_adv.append(max - med)
        max_max2_list_adv.append(max - max2)
        med_min_list_adv.append(med - min)
        max_mean_list_adv.append(max - mean)
        mean_min_list_adv.append(mean - min)

        adv_pred_prob.append(np.mean(end_list))
        entropy_list_adv.append(predictive_entropy(end_list))
        kld_list_adv.append(predictive_kld(end_list, number=mc.hparam.n_sampling)[0])
        std_list_adv.append(predictive_std(end_list, number=mc.hparam.n_sampling)[0])

    if true_flase:
        pst_pred_class = []
        adv_pred_class = []
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

        for i in range(len(adv_filenames)):
            if pst_pred_prob[i] > 0.5:
                pst_pred_class.append(1)
            else:
                pst_pred_class.append(0)
            if adv_pred_prob[i] > 0.5:
                adv_pred_class.append(1)
            else:
                adv_pred_class.append(0)

        for i in range(len(prist_filenames)):
            if pst_pred_class[i] == 1:

                med_mean_list_true.append([prist_filenames[i], med_mean_list_pst[i]])
                max_med_list_true.append([prist_filenames[i], max_med_list_pst[i]])
                max_max2_list_true.append([prist_filenames[i], max_max2_list_pst[i]])
                med_min_list_true.append([prist_filenames[i], med_min_list_pst[i]])
                max_mean_list_true.append([prist_filenames[i], max_mean_list_pst[i]])
                mean_min_list_true.append([prist_filenames[i], mean_min_list_pst[i]])

                entropy_true.append([prist_filenames[i], entropy_list_pst[i]])
                kld_true.append([prist_filenames[i], kld_list_pst[i][0]])
                std_true.append([prist_filenames[i], std_list_pst[i][0]])
            else:

                med_mean_list_flase.append([prist_filenames[i], med_mean_list_pst[i]])
                max_med_list_flase.append([prist_filenames[i], max_med_list_pst[i]])
                max_max2_list_flase.append([prist_filenames[i], max_max2_list_pst[i]])
                med_min_list_flase.append([prist_filenames[i], med_min_list_pst[i]])
                max_mean_list_flase.append([prist_filenames[i], max_mean_list_pst[i]])
                mean_min_list_flase.append([prist_filenames[i], mean_min_list_pst[i]])

                entropy_flase.append([prist_filenames[i], entropy_list_pst[i]])
                kld_flase.append([prist_filenames[i], kld_list_pst[i][0]])
                std_flase.append([prist_filenames[i], std_list_pst[i][0]])
            if adv_pred_class[i] == 1:
                med_mean_list_true.append([adv_filenames[i], med_mean_list_adv[i]])
                max_med_list_true.append([adv_filenames[i], max_med_list_adv[i]])
                max_max2_list_true.append([adv_filenames[i], max_max2_list_adv[i]])
                med_min_list_true.append([adv_filenames[i], med_min_list_adv[i]])
                max_mean_list_true.append([adv_filenames[i], max_mean_list_adv[i]])
                mean_min_list_true.append([adv_filenames[i], mean_min_list_adv[i]])

                entropy_true.append([adv_filenames[i], entropy_list_adv[i]])
                kld_true.append([adv_filenames[i], kld_list_adv[i][0]])
                std_true.append([adv_filenames[i], std_list_adv[i][0]])
            else:
                med_mean_list_flase.append([adv_filenames[i], med_mean_list_adv[i]])
                max_med_list_flase.append([adv_filenames[i], max_med_list_adv[i]])
                max_max2_list_flase.append([adv_filenames[i], max_max2_list_adv[i]])
                med_min_list_flase.append([adv_filenames[i], med_min_list_adv[i]])
                max_mean_list_flase.append([adv_filenames[i], max_mean_list_adv[i]])
                mean_min_list_flase.append([adv_filenames[i], mean_min_list_adv[i]])

                entropy_flase.append([adv_filenames[i], entropy_list_adv[i]])
                kld_flase.append([adv_filenames[i], kld_list_adv[i][0]])
                std_flase.append([adv_filenames[i], std_list_adv[i][0]])

        save_std_path_true = csv_save_path + "std/true_mcdropout_std_" + epoch + ".csv"
        save_std_path_flase = csv_save_path + "std/flase_mcdropout_std_" + epoch + ".csv"
        save_ent_path_true = csv_save_path + "entropy/true_mcdropout_entropy_" + epoch + ".csv"
        save_ent_path_flase = csv_save_path + "entropy/flase_mcdropout_entropy_" + epoch + ".csv"
        save_kld_path_true = csv_save_path + "kld/true_mcdropout_kld_" + epoch + ".csv"
        save_kld_path_flase = csv_save_path + "kld/flase_mcdropout_kld_" + epoch + ".csv"

        pd.DataFrame(entropy_true).to_csv(save_ent_path_true)
        pd.DataFrame(entropy_flase).to_csv(save_ent_path_flase)
        pd.DataFrame(kld_true).to_csv(save_kld_path_true)
        pd.DataFrame(kld_flase).to_csv(save_kld_path_flase)
        pd.DataFrame(std_true).to_csv(save_std_path_true)
        pd.DataFrame(std_flase).to_csv(save_std_path_flase)

        save_med_min_true = csv_save_path + "med_min/true_mcdropout_med_min_" + epoch + ".csv"
        save_max_med_true = csv_save_path + "max_med/true_mcdropout_max_med_" + epoch + ".csv"
        save_med_mean_true = csv_save_path + "med_mean/true_mcdropout_med_mean_" + epoch + ".csv"
        save_max_max2_true = csv_save_path + "max_max2/true_mcdropout_max_max2_" + epoch + ".csv"
        save_max_mean_true = csv_save_path + "max_mean/true_mcdropout_max_mean_" + epoch + ".csv"
        save_mean_min_true = csv_save_path + "mean_min/true_mcdropout_mean_min_" + epoch + ".csv"
        save_med_min_flase = csv_save_path + "med_min/flase_mcdropout_med_min_" + epoch + ".csv"
        save_max_med_flase = csv_save_path + "max_med/flase_mcdropout_max_med_" + epoch + ".csv"
        save_med_mean_flase = csv_save_path + "med_mean/flase_mcdropout_med_mean_" + epoch + ".csv"
        save_max_max2_flase = csv_save_path + "max_max2/flase_mcdropout_max_max2_" + epoch + ".csv"
        save_max_mean_flase = csv_save_path + "max_mean/flase_mcdropout_max_mean_" + epoch + ".csv"
        save_mean_min_flase = csv_save_path + "mean_min/flase_mcdropout_mean_min_" + epoch + ".csv"

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

        save_std_path_all = csv_save_path + "all/mcdropout_std_" + epoch + ".csv"
        save_ent_path_all = csv_save_path + "all/mcdropout_ent_" + epoch + ".csv"
        save_kld_path_all = csv_save_path + "all/mcdropout_kld_" + epoch + ".csv"
        pd.DataFrame(entropy_true + entropy_flase).to_csv(save_ent_path_all)
        pd.DataFrame(kld_true + kld_flase).to_csv(save_kld_path_all)
        pd.DataFrame(std_true + std_flase).to_csv(save_std_path_all)

        save_max_mean_path_all = csv_save_path + "all/mcdropout_max_mean_" + epoch + ".csv"
        save_max_max2_path_all = csv_save_path + "all/mcdropout_max_max2_" + epoch + ".csv"
        save_med_mean_path_all = csv_save_path + "all/mcdropout_med_mean_" + epoch + ".csv"
        save_med_min_path_all = csv_save_path + "all/mcdropout_med_min_" + epoch + ".csv"
        save_max_med_path_all = csv_save_path + "all/mcdropout_max_med_" + epoch + ".csv"
        save_mean_min_path_all = csv_save_path + "all/mcdropout_mean_min_" + epoch + ".csv"

        pd.DataFrame(max_max2_list_true + max_max2_list_flase).to_csv(save_max_max2_path_all)
        pd.DataFrame(max_med_list_true + max_med_list_flase).to_csv(save_max_med_path_all)
        pd.DataFrame(max_mean_list_true + max_med_list_flase).to_csv(save_max_mean_path_all)
        pd.DataFrame(mean_min_list_true + mean_min_list_flase).to_csv(save_mean_min_path_all)
        pd.DataFrame(med_mean_list_true + med_mean_list_flase).to_csv(save_med_mean_path_all)
        pd.DataFrame(med_min_list_true + med_min_list_flase).to_csv(save_med_min_path_all)

        return entropy_true, entropy_flase, kld_true, kld_flase, std_true, std_flase
    else:
        pst_std_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/std/pst_mcdropout_std_" + epoch + ".txt"
        adv_std_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/std/adv_mcdropout_std_" + epoch + ".txt"
        pst_kld_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/kld/pst_mcdropout_kld_" + epoch + ".txt"
        adv_kld_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/kld/adv_mcdropout_kld_" + epoch + ".txt"
        pst_ent_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/entropy/pst_mcdropout_entropy_" + epoch + ".txt"
        adv_ent_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/entropy/adv_mcdropout_entropy_" + epoch + ".txt"
        np.savetxt(pst_std_file_name, std_list_pst)
        np.savetxt(adv_std_file_name, std_list_adv)
        np.savetxt(pst_kld_file_name, kld_list_pst)
        np.savetxt(adv_kld_file_name, kld_list_adv)
        np.savetxt(pst_ent_file_name, entropy_list_pst)
        np.savetxt(adv_ent_file_name, entropy_list_adv)

        adv_med_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_min/adv_mcdropout_med_min_" + epoch + ".txt"
        pst_med_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_min/pst_mcdropout_med_min_" + epoch + ".txt"
        adv_max_med_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_med/adv_mcdropout_max_med_" + epoch + ".txt"
        pst_max_med_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_med/pst_mcdropout_max_med_" + epoch + ".txt"
        adv_med_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_mean/adv_mcdropout_med_mean_" + epoch + ".txt"
        pst_med_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_mean/pst_mcdropout_med_mean_" + epoch + ".txt"
        adv_max_max2_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_max2/adv_mcdropout_max_max2_" + epoch + ".txt"
        pst_max_max2_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_max2/pst_mcdropout_max_max2_" + epoch + ".txt"
        adv_max_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_mean/adv_mcdropout_max_mean_" + epoch + ".txt"
        pst_max_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_mean/pst_mcdropout_max_mean_" + epoch + ".txt"
        adv_mean_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/mean_min/adv_mcdropout_mean_min_" + epoch + ".txt"
        pst_mean_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/mean_min/pst_mcdropout_mean_min_" + epoch + ".txt"

        np.savetxt(adv_med_min_file_name, med_min_list_adv)
        np.savetxt(pst_med_min_file_name, med_min_list_pst)
        np.savetxt(adv_max_med_file_name, max_med_list_adv)
        np.savetxt(pst_max_med_file_name, max_med_list_pst)
        np.savetxt(adv_med_mean_file_name, med_mean_list_adv)
        np.savetxt(pst_med_mean_file_name, med_mean_list_pst)
        np.savetxt(adv_max_max2_file_name, max_max2_list_adv)
        np.savetxt(pst_max_max2_file_name, max_max2_list_pst)
        np.savetxt(adv_max_mean_file_name, max_mean_list_adv)
        np.savetxt(pst_max_mean_file_name, max_mean_list_pst)
        np.savetxt(adv_mean_min_file_name, mean_min_list_adv)
        np.savetxt(pst_mean_min_file_name, mean_min_list_pst)
        return entropy_list_pst, entropy_list_adv, kld_list_pst, kld_list_adv, std_list_pst, std_list_adv, \
               med_min_list_adv, max_med_list_adv, med_min_list_pst, max_med_list_pst, med_mean_list_pst, \
               med_mean_list_adv, max_max2_list_pst, max_max2_list_adv, max_mean_list_pst, max_mean_list_adv, \
               mean_min_list_pst, mean_min_list_adv


def get_deep_ensemble_uc(prist_data, adv_data, model_directory, epoch, n_members, prist_filenames, adv_filenames,
                         architecture_type="dnn",
                         true_flase=False):
    csv_save_path = "/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/"
    d_ensemble = DeepEnsemble(architecture_type=architecture_type, n_members=n_members,
                              model_directory=model_directory)
    d_ensemble_pst_pred = d_ensemble.predict(prist_data)
    d_ensemble_adv_pred = d_ensemble.predict(adv_data)

    entropy_list_adv = []
    entropy_list_pst = []
    kld_list_adv = []
    kld_list_pst = []
    std_list_adv = []
    std_list_pst = []
    pst_pred_prob = []
    adv_pred_prob = []

    med_min_list_adv = []
    max_med_list_adv = []
    med_min_list_pst = []
    max_med_list_pst = []
    med_mean_list_pst = []
    med_mean_list_adv = []
    max_max2_list_pst = []
    max_max2_list_adv = []
    max_mean_list_pst = []
    max_mean_list_adv = []
    mean_min_list_pst = []
    mean_min_list_adv = []

    for item in d_ensemble_pst_pred:
        end_list = []
        for i in range(n_members):
            end_list.append(item[i][0])
        max2 = np.sort(end_list)[-2]
        mean = np.mean(end_list)
        med = np.median(end_list)
        max = np.max(end_list)
        min = np.min(end_list)

        med_mean_list_pst.append(med - mean)
        max_med_list_pst.append(max - med)
        max_max2_list_pst.append(max - max2)
        med_min_list_pst.append(med - min)
        max_mean_list_pst.append(max - mean)
        mean_min_list_pst.append(mean - min)

        pst_pred_prob.append(np.mean(end_list))
        entropy_list_pst.append(predictive_entropy(end_list))
        kld_list_pst.append(predictive_kld(end_list, number=n_members)[0])
        std_list_pst.append(predictive_std(end_list, number=n_members)[0])
    for item in d_ensemble_adv_pred:
        end_list = []
        for i in range(n_members):
            end_list.append(item[i][0])
        max2 = np.sort(end_list)[-2]
        mean = np.mean(end_list)
        med = np.median(end_list)
        max = np.max(end_list)
        min = np.min(end_list)

        med_mean_list_adv.append(med - mean)
        max_med_list_adv.append(max - med)
        max_max2_list_adv.append(max - max2)
        med_min_list_adv.append(med - min)
        max_mean_list_adv.append(max - mean)
        mean_min_list_adv.append(mean - min)

        adv_pred_prob.append(np.mean(end_list))
        entropy_list_adv.append(predictive_entropy(end_list))
        kld_list_adv.append(predictive_kld(end_list, number=n_members)[0])
        std_list_adv.append(predictive_std(end_list, number=n_members)[0])

    if true_flase:
        pst_pred_class = []
        adv_pred_class = []
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

        for i in range(len(adv_filenames)):
            if pst_pred_prob[i] > 0.5:
                pst_pred_class.append(1)
            else:
                pst_pred_class.append(0)
            if adv_pred_prob[i] > 0.5:
                adv_pred_class.append(1)
            else:
                adv_pred_class.append(0)

        for i in range(len(prist_filenames)):
            if pst_pred_class[i] == 1:
                med_mean_list_true.append([prist_filenames[i], med_mean_list_pst[i]])
                max_med_list_true.append([prist_filenames[i], max_med_list_pst[i]])
                max_max2_list_true.append([prist_filenames[i], max_max2_list_pst[i]])
                med_min_list_true.append([prist_filenames[i], med_min_list_pst[i]])
                max_mean_list_true.append([prist_filenames[i], max_mean_list_pst[i]])
                mean_min_list_true.append([prist_filenames[i], mean_min_list_pst[i]])

                entropy_true.append([prist_filenames[i], entropy_list_pst[i]])
                kld_true.append([prist_filenames[i], kld_list_pst[i][0]])
                std_true.append([prist_filenames[i], std_list_pst[i][0]])
            else:
                med_mean_list_flase.append([prist_filenames[i], med_mean_list_pst[i]])
                max_med_list_flase.append([prist_filenames[i], max_med_list_pst[i]])
                max_max2_list_flase.append([prist_filenames[i], max_max2_list_pst[i]])
                med_min_list_flase.append([prist_filenames[i], med_min_list_pst[i]])
                max_mean_list_flase.append([prist_filenames[i], max_mean_list_pst[i]])
                mean_min_list_flase.append([prist_filenames[i], mean_min_list_pst[i]])

                entropy_flase.append([prist_filenames[i], entropy_list_pst[i]])
                kld_flase.append([prist_filenames[i], kld_list_pst[i][0]])
                std_flase.append([prist_filenames[i], std_list_pst[i][0]])
            if adv_pred_class[i] == 1:
                med_mean_list_true.append([adv_filenames[i], med_mean_list_adv[i]])
                max_med_list_true.append([adv_filenames[i], max_med_list_adv[i]])
                max_max2_list_true.append([adv_filenames[i], max_max2_list_adv[i]])
                med_min_list_true.append([adv_filenames[i], med_min_list_adv[i]])
                max_mean_list_true.append([adv_filenames[i], max_mean_list_adv[i]])
                mean_min_list_true.append([adv_filenames[i], mean_min_list_adv[i]])

                entropy_true.append([adv_filenames[i], entropy_list_adv[i]])
                kld_true.append([adv_filenames[i], kld_list_adv[i][0]])
                std_true.append([adv_filenames[i], std_list_adv[i][0]])
            else:
                med_mean_list_flase.append([adv_filenames[i], med_mean_list_adv[i]])
                max_med_list_flase.append([adv_filenames[i], max_med_list_adv[i]])
                max_max2_list_flase.append([adv_filenames[i], max_max2_list_adv[i]])
                med_min_list_flase.append([adv_filenames[i], med_min_list_adv[i]])
                max_mean_list_flase.append([adv_filenames[i], max_mean_list_adv[i]])
                mean_min_list_flase.append([adv_filenames[i], mean_min_list_adv[i]])

                entropy_flase.append([adv_filenames[i], entropy_list_adv[i]])
                kld_flase.append([adv_filenames[i], kld_list_adv[i][0]])
                std_flase.append([adv_filenames[i], std_list_adv[i][0]])

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
    else:
        pst_kld_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/kld/pst_deepensemble_kld_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_kld_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/kld/adv_deepensemble_kld_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_ent_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/entropy/pst_deepensemble_entropy_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_ent_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/entropy/adv_deepensemble_entropy_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_std_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/std/pst_deepensemble_std_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_std_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/std/adv_deepensemble_std_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        np.savetxt(pst_std_file_name, std_list_pst)
        np.savetxt(adv_std_file_name, std_list_adv)
        np.savetxt(pst_ent_file_name, entropy_list_pst)
        np.savetxt(adv_ent_file_name, entropy_list_adv)
        np.savetxt(pst_kld_file_name, kld_list_pst)
        np.savetxt(adv_kld_file_name, kld_list_adv)

        adv_med_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_min/adv_deepensemble_med_min_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_med_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_min/pst_deepensemble_med_min_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_max_med_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_med/adv_deepensemble_max_med_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_max_med_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_med/pst_deepensemble_max_med_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_med_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_mean/adv_deepensemble_med_mean_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_med_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_mean/pst_deepensemble_med_mean_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_max_max2_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_max2/adv_deepensemble_max_max2_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_max_max2_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_max2/pst_deepensemble_max_max2_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_max_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_mean/adv_deepensemble_max_mean_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_max_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_mean/pst_deepensemble_max_mean_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_mean_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/mean_min/adv_deepensemble_mean_min_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_mean_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/mean_min/pst_deepensemble_mean_min_" + str(
            epoch) + "_" + str(n_members) + ".txt"

        np.savetxt(adv_med_min_file_name, med_min_list_adv)
        np.savetxt(pst_med_min_file_name, med_min_list_pst)
        np.savetxt(adv_max_med_file_name, max_med_list_adv)
        np.savetxt(pst_max_med_file_name, max_med_list_pst)
        np.savetxt(adv_med_mean_file_name, med_mean_list_adv)
        np.savetxt(pst_med_mean_file_name, med_mean_list_pst)
        np.savetxt(adv_max_max2_file_name, max_max2_list_adv)
        np.savetxt(pst_max_max2_file_name, max_max2_list_pst)
        np.savetxt(adv_max_mean_file_name, max_mean_list_adv)
        np.savetxt(pst_max_mean_file_name, max_mean_list_pst)
        np.savetxt(adv_mean_min_file_name, mean_min_list_adv)
        np.savetxt(pst_mean_min_file_name, mean_min_list_pst)
        return entropy_list_pst, entropy_list_adv, kld_list_pst, kld_list_adv, std_list_pst, std_list_adv, \
               med_min_list_adv, max_med_list_adv, med_min_list_pst, max_med_list_pst, med_mean_list_pst, \
               med_mean_list_adv, max_max2_list_pst, max_max2_list_adv, max_mean_list_pst, max_mean_list_adv, \
               mean_min_list_pst, mean_min_list_adv


def get_weight_ensemble_uc(prist_data, adv_data, model_directory, epoch, n_members, prist_filenames, adv_filenames,architecture_type="dnn",
                           true_flase=False):
    csv_save_path = "/home/lhd/uncertainity_malware/myexperiment/output/uncertainty_metrics/uc_csv/"
    w_ensemble = WeightedDeepEnsemble(architecture_type=architecture_type, n_members=n_members,
                                      model_directory=model_directory)
    w_ensemble_pst_pred = w_ensemble.predict(prist_data)
    w_ensemble_adv_pred = w_ensemble.predict(adv_data)

    entropy_list_adv = []
    entropy_list_pst = []
    kld_list_adv = []
    kld_list_pst = []
    std_list_adv = []
    std_list_pst = []
    pst_pred_prob = []
    adv_pred_prob = []

    med_min_list_adv = []
    max_med_list_adv = []
    med_min_list_pst = []
    max_med_list_pst = []
    med_mean_list_pst = []
    med_mean_list_adv = []
    max_max2_list_pst = []
    max_max2_list_adv = []
    max_mean_list_pst = []
    max_mean_list_adv = []
    mean_min_list_pst = []
    mean_min_list_adv = []

    for item in w_ensemble_pst_pred[0]:
        end_list = []
        for i in range(n_members):
            end_list.append(item[i][0])

        max2 = np.sort(end_list)[-2]
        mean = np.mean(end_list)
        med = np.median(end_list)
        max = np.max(end_list)
        min = np.min(end_list)

        med_mean_list_pst.append(med - mean)
        max_med_list_pst.append(max - med)
        max_max2_list_pst.append(max - max2)
        med_min_list_pst.append(med - min)
        max_mean_list_pst.append(max - mean)
        mean_min_list_pst.append(mean - min)

        pst_pred_prob.append(np.mean(end_list))
        entropy_list_pst.append(predictive_entropy(end_list))
        kld_list_pst.append(predictive_kld(end_list, number=n_members)[0])
        std_list_pst.append(predictive_std(end_list, number=n_members)[0])
    for item in w_ensemble_adv_pred[0]:
        end_list = []
        for i in range(n_members):
            end_list.append(item[i][0])

        max2 = np.sort(end_list)[-2]
        mean = np.mean(end_list)
        med = np.median(end_list)
        max = np.max(end_list)
        min = np.min(end_list)

        med_mean_list_adv.append(med - mean)
        max_med_list_adv.append(max - med)
        max_max2_list_adv.append(max - max2)
        med_min_list_adv.append(med - min)
        max_mean_list_adv.append(max - mean)
        mean_min_list_adv.append(mean - min)

        adv_pred_prob.append(np.mean(end_list))
        entropy_list_adv.append(predictive_entropy(end_list))
        kld_list_adv.append(predictive_kld(end_list, number=n_members)[0])
        std_list_adv.append(predictive_std(end_list, number=n_members)[0])

    if true_flase:
        pst_pred_class = []
        adv_pred_class = []
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

        for i in range(len(adv_filenames)):
            if pst_pred_prob[i] > 0.5:
                pst_pred_class.append(1)
            else:
                pst_pred_class.append(0)
            if adv_pred_prob[i] > 0.5:
                adv_pred_class.append(1)
            else:
                adv_pred_class.append(0)

        for i in range(len(prist_filenames)):
            if pst_pred_class[i] == 1:

                med_mean_list_true.append([prist_filenames[i], med_mean_list_pst[i]])
                max_med_list_true.append([prist_filenames[i], max_med_list_pst[i]])
                max_max2_list_true.append([prist_filenames[i], max_max2_list_pst[i]])
                med_min_list_true.append([prist_filenames[i], med_min_list_pst[i]])
                max_mean_list_true.append([prist_filenames[i], max_mean_list_pst[i]])
                mean_min_list_true.append([prist_filenames[i], mean_min_list_pst[i]])

                entropy_true.append([prist_filenames[i], entropy_list_pst[i]])
                kld_true.append([prist_filenames[i], kld_list_pst[i][0]])
                std_true.append([prist_filenames[i], std_list_pst[i][0]])
            else:

                med_mean_list_flase.append([prist_filenames[i], med_mean_list_pst[i]])
                max_med_list_flase.append([prist_filenames[i], max_med_list_pst[i]])
                max_max2_list_flase.append([prist_filenames[i], max_max2_list_pst[i]])
                med_min_list_flase.append([prist_filenames[i], med_min_list_pst[i]])
                max_mean_list_flase.append([prist_filenames[i], max_mean_list_pst[i]])
                mean_min_list_flase.append([prist_filenames[i], mean_min_list_pst[i]])

                entropy_flase.append([prist_filenames[i], entropy_list_pst[i]])
                kld_flase.append([prist_filenames[i], kld_list_pst[i][0]])
                std_flase.append([prist_filenames[i], std_list_pst[i][0]])
            if adv_pred_class[i] == 1:

                med_mean_list_true.append([adv_filenames[i], med_mean_list_adv[i]])
                max_med_list_true.append([adv_filenames[i], max_med_list_adv[i]])
                max_max2_list_true.append([adv_filenames[i], max_max2_list_adv[i]])
                med_min_list_true.append([adv_filenames[i], med_min_list_adv[i]])
                max_mean_list_true.append([adv_filenames[i], max_mean_list_adv[i]])
                mean_min_list_true.append([adv_filenames[i], mean_min_list_adv[i]])

                entropy_true.append([adv_filenames[i], entropy_list_adv[i]])
                kld_true.append([adv_filenames[i], kld_list_adv[i][0]])
                std_true.append([adv_filenames[i], std_list_adv[i][0]])
            else:

                med_mean_list_flase.append([adv_filenames[i], med_mean_list_adv[i]])
                max_med_list_flase.append([adv_filenames[i], max_med_list_adv[i]])
                max_max2_list_flase.append([adv_filenames[i], max_max2_list_adv[i]])
                med_min_list_flase.append([adv_filenames[i], med_min_list_adv[i]])
                max_mean_list_flase.append([adv_filenames[i], max_mean_list_adv[i]])
                mean_min_list_flase.append([adv_filenames[i], mean_min_list_adv[i]])

                entropy_flase.append([adv_filenames[i], entropy_list_adv[i]])
                kld_flase.append([adv_filenames[i], kld_list_adv[i][0]])
                std_flase.append([adv_filenames[i], std_list_adv[i][0]])

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
    else:
        pst_kld_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/kld/pst_weightensemble_kld_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_kld_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/kld/adv_weightensemble_kld_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_ent_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/entropy/pst_weightensemble_entropy_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_ent_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/entropy/adv_weightensemble_entropy_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_std_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/std/pst_weightensemble_std_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_std_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/std/adv_weightensemble_std_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        np.savetxt(pst_std_file_name, std_list_pst)
        np.savetxt(adv_std_file_name, std_list_adv)
        np.savetxt(pst_ent_file_name, entropy_list_pst)
        np.savetxt(adv_ent_file_name, entropy_list_adv)
        np.savetxt(pst_kld_file_name, kld_list_pst)
        np.savetxt(adv_kld_file_name, kld_list_adv)

        adv_med_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_min/adv_weightensemble_med_min_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_med_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_min/pst_weightensemble_med_min_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_max_med_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_med/adv_weightensemble_max_med_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_max_med_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_med/pst_weightensemble_max_med_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_med_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_mean/adv_weightensemble_med_mean_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_med_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/med_mean/pst_weightensemble_med_mean_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_max_max2_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_max2/adv_weightensemble_max_max2_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_max_max2_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_max2/pst_weightensemble_max_max2_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_max_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_mean/adv_weightensemble_max_mean_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_max_mean_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/max_mean/pst_weightensemble_max_mean_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        adv_mean_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/mean_min/adv_weightensemble_mean_min_" + str(
            epoch) + "_" + str(n_members) + ".txt"
        pst_mean_min_file_name = "/home/lhd/Uncertainity_malware/myexperiment/uc_txt/mean_min/pst_weightensemble_mean_min_" + str(
            epoch) + "_" + str(n_members) + ".txt"

        np.savetxt(adv_med_min_file_name, med_min_list_adv)
        np.savetxt(pst_med_min_file_name, med_min_list_pst)
        np.savetxt(adv_max_med_file_name, max_med_list_adv)
        np.savetxt(pst_max_med_file_name, max_med_list_pst)
        np.savetxt(adv_med_mean_file_name, med_mean_list_adv)
        np.savetxt(pst_med_mean_file_name, med_mean_list_pst)
        np.savetxt(adv_max_max2_file_name, max_max2_list_adv)
        np.savetxt(pst_max_max2_file_name, max_max2_list_pst)
        np.savetxt(adv_max_mean_file_name, max_mean_list_adv)
        np.savetxt(pst_max_mean_file_name, max_mean_list_pst)
        np.savetxt(adv_mean_min_file_name, mean_min_list_adv)
        np.savetxt(pst_mean_min_file_name, mean_min_list_pst)
        return entropy_list_pst, entropy_list_adv, kld_list_pst, kld_list_adv, std_list_pst, std_list_adv, \
               med_min_list_adv, max_med_list_adv, med_min_list_pst, max_med_list_pst, med_mean_list_pst, \
               med_mean_list_adv, max_max2_list_pst, max_max2_list_adv, max_mean_list_pst, max_mean_list_adv, \
               mean_min_list_pst, mean_min_list_adv
