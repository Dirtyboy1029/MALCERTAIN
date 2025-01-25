# -*- coding: utf-8 -*- 
# @Time : 2024/2/13 10:50 
# @Author : DirtyBoy 
# @File : metrics_utils.py
import numpy as np


def _check_probablities(p, q=None):
    assert 0. <= np.all(p) <= 1.
    if q is not None:
        assert len(p) == len(q), \
            'Probabilies and ground truth must have the same number of elements.'


def max_max2(end_list):
    max2 = np.sort(end_list)[-2]
    max = np.max(end_list)
    return max - max2


def min2_min(end_list):
    min2 = np.sort(end_list)[1]
    min = np.min(end_list)
    return min2 - min


def max_min(end_list):
    min = np.min(end_list)
    max = np.max(end_list)
    return max - min


def mean_med(end_list):
    mean = np.mean(end_list)
    med = np.median(end_list)
    return med - mean


def predictive_entropy(p, number=10, w=None, base=2, eps=1e-10):
    """
    计算子模型预测结果的不确定性熵
    :param base: 对数的基，默认为 2
    :param eps: 小值，防止 log(0)
    :return: 子模型预测分布的熵
    """
    predictions = np.asarray(p)
    if w is None:
        weights = np.ones(shape=(number, 1), dtype=float) / number
    else:
        weights = np.asarray(w)
    weights = weights / np.sum(weights)
    entropy = -np.sum(weights * (
            predictions * np.log(predictions + eps) +
            (1 - predictions) * np.log(1 - predictions + eps)
    ))
    if base is not None:
        entropy /= np.log(base)

    return entropy


def predictive_kld(p, number=10, w=None, base=2, eps=1e-10):
    """
    The Kullback-Leibler (KL) divergence measures the difference between two probability distributions by quantifying the information lost
    when one distribution is approximated by another. When comparing a probability vector to its mean vector, the KL divergence assesses
    the information difference between the original probabilities and the uniform distribution of their mean.


    calculate Kullback-Leibler divergence in element-wise
    :param p: probabilities
    :param number: the number of likelihood values for each sample
    :param w: weights for probabilities
    :param base: default exp
    :return: average entropy value
    """
    if number <= 1:
        return np.zeros_like(p)

    p_arr = np.asarray(p).reshape((-1, number))
    _check_probablities(p)
    q_arr = np.tile(np.mean(p_arr, axis=-1, keepdims=True), [1, number])
    if w is None:
        w_arr = np.ones(shape=(number, 1), dtype=float) / number
    else:
        w_arr = np.asarray(w).reshape((number, 1))

    kld_elem = p_arr * np.log((p_arr + eps) / (q_arr + eps)) + (1. - p_arr) * np.log(
        (1. - p_arr + eps) / (1. - q_arr + eps))
    if base is not None:
        kld_elem = kld_elem / np.log(base)
    kld = np.matmul(kld_elem, w_arr)
    return kld[0][0]


def predictive_std(p, number=10, w=None):
    """
    calculate the probabilities deviation
    :param p: probabilities
    :param number: the number of probabilities applied to each sample
    :param w: weights for probabilities
    :param axis: the axis along which the calculation is conducted
    :return:
    """
    if number <= 1:
        return np.zeros_like(p)

    ps_arr = np.asarray(p).reshape((-1, number))
    _check_probablities(ps_arr)
    if w is None:
        w = np.ones(shape=(number, 1), dtype=float) / number
    else:
        w = np.asarray(w).reshape((number, 1))
    assert 0 <= np.all(w) <= 1.
    mean = np.matmul(ps_arr, w)
    var = np.sqrt(np.matmul(np.square(ps_arr - mean), w) * (float(number) / float(number - 1)))
    return var[0][0]


def nll(p, label, eps=1e-10, base=2):
    """
    negative log likelihood (NLL)
    :param p: predictive labels
    :param eps: a small value prevents the overflow
    :param base: the base of log function
    :return: the mean of NLL
    """
    p = np.array(p)
    q = np.full(len(p), label)
    nll = -(q * np.log(p + eps) + (1. - q) * np.log(1. - p + eps))
    if base is not None:
        nll = np.clip(nll / np.log(base), a_min=0., a_max=1000)
    return np.mean(nll)


def prob_label_kld(p, label, number=10, w=None, base=2, eps=1e-10):
    if number <= 1:
        return np.zeros_like(p)

    p_arr = np.asarray(p).reshape((-1, number))
    _check_probablities(p)
    q_arr = np.full(number, label)
    if w is None:
        w_arr = np.ones(shape=(number, 1), dtype=float) / number
    else:
        w_arr = np.asarray(w).reshape((number, 1))

    kld_elem = p_arr * np.log((p_arr + eps) / (q_arr + eps)) + (1. - p_arr) * np.log(
        (1. - p_arr + eps) / (1. - q_arr + eps))
    if base is not None:
        kld_elem = kld_elem / np.log(base)
    kld = np.matmul(kld_elem, w_arr)

    return (kld / number)[0][0]


def Wasserstein_distance(p, label):
    from scipy.stats import wasserstein_distance
    p = np.array(p)
    q = np.full(len(p), label)
    emd = wasserstein_distance(p, q)

    return emd


def Euclidean_distance(p, label):
    p = np.array(p)
    q = np.full(len(p), label)

    v1 = np.array(p)
    v2 = np.array(q)

    distance = np.linalg.norm(v1 - v2)

    return distance


def Manhattan_distance(p, label):
    p = np.array(p)
    q = np.full(len(p), label)
    v1 = np.array(p)
    v2 = np.array(q)
    distance = np.sum(np.abs(v1 - v2)) / len(p)

    return distance


def Chebyshev_distance(p, label):
    p = np.array(p)
    q = np.full(len(p), label)
    v1 = np.array(p)
    v2 = np.array(q)
    distance = np.max(np.abs(v1 - v2))

    return distance
