# -*- coding: utf-8 -*- 
# @Time : 2024/9/24 10:27 
# @Author : DirtyBoy 
# @File : utils.py
import numpy as np
import os, joblib
from metrics_utils import *
metrics_dict = {
    'entropy': predictive_entropy,
    'kld': predictive_kld,
    'std': predictive_std,
    'max_max2': max_max2,
    'min2_min': min2_min,
    'max_min': max_min,
    'mean_med': mean_med
}

def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def evaluate(x_prob, gt_labels, threshold=0.5, name='test'):
    """
    get some statistical values
    :param gt_labels: ground truth labels
    :param threshold: float value between 0 and 1, to decide the predicted label
    :return: None
    """
    x_pred = (x_prob >= threshold).astype(np.int32)

    # metrics
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
    accuracy = accuracy_score(gt_labels, x_pred)
    b_accuracy = balanced_accuracy_score(gt_labels, x_pred)

    MSG = "The accuracy on the {} dataset is {:.5f}%"
    print(MSG.format(name, accuracy * 100))
    MSG = "The balanced accuracy on the {} dataset is {:.5f}%"
    print(MSG.format(name, b_accuracy * 100))
    is_single_class = False
    if np.all(gt_labels == 1.) or np.all(gt_labels == 0.):
        is_single_class = True
    if not is_single_class:
        tn, fp, fn, tp = confusion_matrix(gt_labels, x_pred).ravel()

        fpr = fp / float(tn + fp)
        fnr = fn / float(tp + fn)
        f1 = f1_score(gt_labels, x_pred, average='binary')

        print("Other evaluation metrics we may need:")
        MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%"
        print(MSG.format(fnr * 100, fpr * 100, f1 * 100))
        return MSG.format(fnr * 100, fpr * 100, f1 * 100) + "The balanced accuracy on the {} dataset is {:.5f}%".format(
            name, accuracy * 100)


def evaluate_correction(x_prob, gt_labels, is_false, only_fn=False, name='test'):
    """
    get some statistical values
    :param gt_labels: ground truth labels
    :return: None
    """
    x_pred = (x_prob >= 0.5).astype(np.int32)
    if only_fn:
        for i, item in enumerate(is_false):
            if item == 1 and x_pred[i] == 0:
                x_pred[i] = (x_pred[i] + 1) % 2
    else:
        for i, item in enumerate(is_false):
            if item == 1:
                x_pred[i] = (x_pred[i] + 1) % 2
    # metrics
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
    accuracy = accuracy_score(gt_labels, x_pred)
    b_accuracy = balanced_accuracy_score(gt_labels, x_pred)

    MSG = "The accuracy on the {} dataset is {:.5f}%"
    print(MSG.format(name, accuracy * 100))
    MSG = "The balanced accuracy on the {} dataset is {:.5f}%"
    print(MSG.format(name, b_accuracy * 100))
    is_single_class = False
    if np.all(gt_labels == 1.) or np.all(gt_labels == 0.):
        is_single_class = True
    if not is_single_class:
        tn, fp, fn, tp = confusion_matrix(gt_labels, x_pred).ravel()

        Recall = tp / (tp + fn)
        Specificity = tn / (tn + fp)

        IN = Recall + Specificity - 1

        precision = tp / (tp + fp)
        NPV = tn / (tn + fn)

        MK = precision + NPV - 1
        f1 = f1_score(gt_labels, x_pred, average='binary')

        print("Other evaluation metrics we may need:")
        MSG = "Informedness is {:.5f}%, Markedness is {:.5f}%, F1 score is {:.5f}%"
        print(MSG.format(IN * 100, MK * 100, f1 * 100))


def correct_model_training_data_processing(feature_type, data_type, test_data_type):
    vanilla_prob = np.squeeze(
        np.mean(np.load(
            '../Training/output/' + feature_type + '/' + data_type + '_vanilla_' + test_data_type + '.npy'),
            axis=1))
    save_path = os.path.join('../Training/config',
                             'databases_' + feature_type + '_' + test_data_type + '.conf')
    data_filenames, gt_labels = read_joblib(save_path)
    vanilla_pred = (vanilla_prob >= 0.5).astype(np.int32)

    true_index = np.where(vanilla_pred == gt_labels)[0]
    false_index = np.where(vanilla_pred != gt_labels)[0]

    feature = np.zeros((len(data_filenames), 21))
    #feature[:, 0] = vanilla_prob
    for i, model_type in enumerate(['bayesian', 'mc_dropout', 'ensemble']):
        model_output = np.load(
            '../Training/output/' + feature_type + '/' + data_type + '_' + model_type + '_' + test_data_type + '.npy')
        feature[:, int(i * 7 + 1)] = np.squeeze(
            np.mean(model_output, axis=1))

        feature[:, int(i * 7 + 2)] = np.array([predictive_entropy(prob_) for prob_ in model_output])
        feature[:, int(i * 7 + 3)] = np.array([predictive_kld(prob_) for prob_ in model_output])
        feature[:, int(i * 7 + 4)] = np.array([predictive_std(prob_) for prob_ in model_output])

        feature[:, int(i * 7 + 5)] = np.squeeze(np.array([max_max2(prob_) for prob_ in model_output]))
        feature[:, int(i * 7 + 6)] = np.squeeze(np.array([min2_min(prob_) for prob_ in model_output]))
        feature[:, int(i * 7 + 0)] = np.squeeze(np.array([mean_med(prob_) for prob_ in model_output]))

    return feature, list(true_index), list(false_index)


def save_correction_model(model, path):
    joblib.dump(model, path)
    print('save model to ' + path)


def load_correction_model(path):
    return joblib.load(path)
