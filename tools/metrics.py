import numpy as np


def _check_probablities(p, q=None):
    assert 0. <= np.all(p) <= 1.
    if q is not None:
        assert len(p) == len(q), \
            'Probabilies and ground truth must have the same number of elements.'


def predictive_entropy(p, base=2, eps=1e-10, number=10):
    """
    calculate entropy in element-wise
    :param p: probabilities
    :param base: default exp
    :return: average entropy value
    """
    p_arr = np.asarray(p)
    _check_probablities(p)
    enc = -(p_arr * np.log(p_arr + eps) + (1. - p_arr) * np.log(1. - p_arr + eps))
    if base is not None:
        enc = np.clip(enc / np.log(base), a_min=0., a_max=1000)
    enc_ = []
    for item in enc:
        enc_.append([np.sum(item)/ number])
    return np.array(enc_)


def entropy(p, number=None, base=2, eps=1e-10):
    """
    calculate entropy in element-wise
    :param p: probabilities
    :param base: default exp
    :return: average entropy value
    """
    p_arr = np.asarray(p)
    _check_probablities(p)
    enc = -(p_arr * np.log(p_arr + eps) + (1. - p_arr) * np.log(1. - p_arr + eps))
    enc = np.sum(enc)
    return enc


def predictive_kld(p, number, w=None, base=2, eps=1e-10):
    """
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
        w_arr = np.ones(shape=(number, 1), dtype=np.float) / number
    else:
        w_arr = np.asarray(w).reshape((number, 1))

    kld_elem = p_arr * np.log((p_arr + eps) / (q_arr + eps)) + (1. - p_arr) * np.log(
        (1. - p_arr + eps) / (1. - q_arr + eps))
    if base is not None:
        kld_elem = kld_elem / np.log(base)
    kld = np.matmul(kld_elem, w_arr)
    return kld


def predictive_std(p, number, w=None):
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
        w = np.ones(shape=(number, 1), dtype=np.float) / number
    else:
        w = np.asarray(w).reshape((number, 1))
    assert 0 <= np.all(w) <= 1.
    mean = np.matmul(ps_arr, w)
    var = np.sqrt(np.matmul(np.square(ps_arr - mean), w) * (float(number) / float(number - 1)))
    return var


def nll(p, q, eps=1e-10, base=2):
    """
    negative log likelihood (NLL)
    :param p: predictive labels
    :param q: ground truth labels
    :param eps: a small value prevents the overflow
    :param base: the base of log function
    :return: the mean of NLL
    """
    _check_probablities(p, q)
    nll = -(q * np.log(p + eps) + (1. - q) * np.log(1. - p + eps))
    if base is not None:
        nll = np.clip(nll / np.log(base), a_min=0., a_max=1000)
    return np.mean(nll)


def b_nll(p, q, eps=1e-10, base=2):
    """
    balanced negative log likelihood (NLL)
    :param p: 1-D array, predictive labels
    :param q: 1-D array, ground truth labels
    :param eps: a small value prevents the overflow
    :param base: the base of log function
    :return: the mean of NLL
    """
    _check_probablities(p, q)
    pos_indicator = (q == 1)
    pos_nll = nll(p[pos_indicator], q[pos_indicator], eps, base)
    neg_indicator = (q == 0)
    neg_nll = nll(p[neg_indicator], q[neg_indicator], eps, base)
    return 1. / 2. * (pos_nll + neg_nll)


def brier_score(p, q, pos_label=1):
    """
    brier score
    :param p: predictive labels
    :param q: ground truth labels
    :param pos_label: the positive class
    :return:
    """
    from sklearn.metrics import brier_score_loss
    _check_probablities(p, q)
    return brier_score_loss(q, p, pos_label=pos_label)


def b_brier_score(p, q):
    """
    balanced brier score
    :param p: predictive labels
    :param q: ground truth labels
    :return:
    """
    pos_indicator = (q == 1)
    pos_bs = brier_score(p[pos_indicator], q[pos_indicator], pos_label=None)
    neg_indicator = (q == 0)
    neg_bs = brier_score(p[neg_indicator], q[neg_indicator], pos_label=None)
    return 1. / 2. * (pos_bs + neg_bs)


def expected_calibration_error(probabilities, ground_truth, bins=10, use_unweighted_version=True):
    """
    Code is adapted from https://github.com/google-research/google-research/tree/master/uq_benchmark_2019/metrics_lib.py
    Compute the expected calibration error of a set of preditions in [0, 1].
    Args:
    probabilities: A numpy vector of N probabilities assigned to each prediction
    ground_truth: A numpy vector of N ground truth labels in {0,1, True, False}
    bins: Number of equal width bins to bin predictions into in [0, 1], or
      an array representing bin edges.
    Returns:
    Float: the expected calibration error.
    """

    def bin_predictions_and_accuracies(probabilities, ground_truth, bins=10):
        """A helper function which histograms a vector of probabilities into bins.

        Args:
          probabilities: A numpy vector of N probabilities assigned to each prediction
          ground_truth: A numpy vector of N ground truth labels in {0,1}
          bins: Number of equal width bins to bin predictions into in [0, 1], or an
            array representing bin edges.

        Returns:
          bin_edges: Numpy vector of floats containing the edges of the bins
            (including leftmost and rightmost).
          accuracies: Numpy vector of floats for the average accuracy of the
            predictions in each bin.
          counts: Numpy vector of ints containing the number of examples per bin.
        """
        _check_probablities(probabilities, ground_truth)

        if isinstance(bins, int):
            num_bins = bins
        else:
            num_bins = bins.size - 1

        # Ensure probabilities are never 0, since the bins in np.digitize are open on
        # one side.
        probabilities = np.where(probabilities == 0, 1e-8, probabilities)
        counts, bin_edges = np.histogram(probabilities, bins=bins, range=[0., 1.])
        indices = np.digitize(probabilities, bin_edges, right=True)
        accuracies = np.array([np.mean(ground_truth[indices == i])
                               for i in range(1, num_bins + 1)])
        return bin_edges, accuracies, counts

    def bin_centers_of_mass(probabilities, bin_edges):
        probabilities = np.where(probabilities == 0, 1e-8, probabilities)
        indices = np.digitize(probabilities, bin_edges, right=True)
        return np.array([np.mean(probabilities[indices == i])
                         for i in range(1, len(bin_edges))])

    probabilities = probabilities.flatten()
    ground_truth = ground_truth.flatten()

    bin_edges, accuracies, counts = bin_predictions_and_accuracies(
        probabilities, ground_truth, bins)
    bin_centers = bin_centers_of_mass(probabilities, bin_edges)
    num_examples = np.sum(counts)
    if not use_unweighted_version:
        ece = np.sum([(counts[i] / float(num_examples)) * np.sum(
            np.abs(bin_centers[i] - accuracies[i]))
                      for i in range(bin_centers.size) if counts[i] > 0])
    else:
        ece = np.sum([(1. / float(bins)) * np.sum(
            np.abs(bin_centers[i] - accuracies[i]))
                      for i in range(bin_centers.size) if counts[i] > 0])
    return ece


def _main():
    gt_labels = np.random.choice(2, (1000,))
    prob = np.random.uniform(0., 1., (1000,))
    print(gt_labels)
    print(prob)
    print("Entropy:", entropy1(prob))
    # print("negative log likelihood:", nll(prob, gt_labels))
    # print('balanced nll:', b_nll(prob, gt_labels))
    # print("brier score:", brier_score(prob, gt_labels))
    # print("balanced brier score:", b_brier_score(prob, gt_labels))
    # print("expected calibration error:", expected_calibration_error(prob, gt_labels))
    #
    # prob2 = np.random.uniform(0., 1., (1000, 10))
    # w = np.random.uniform(0., 1., (10,))
    # w = w/np.sum(w)
    # print("Standard deviation:", predictive_std(prob2, weights=w, number=10))
    # print("Kl divergence:", predictive_kld(prob2, w, number=10))
    # print("Standard deviation:", predictive_std(np.zeros(shape=(10, 10)), weights=w, number=10))
    # print("Kl divergence:", predictive_kld(np.zeros(shape=(10, 10)), w, number=10))

    gt_labels = np.array([1., 1., 0., 0, 0])
    prob = gt_labels
    # print("Entropy:", entropy(prob))
    # print("negative log likelihood:", nll(prob, gt_labels))
    # print('balanced nll:', b_nll(prob, gt_labels))

    # print("Standard deviation:", predictive_std(prob, number=2))
    # print("Kl divergence:", predictive_kld(prob, number=2))
    # print("brier score:", brier_score(prob, gt_labels))
    # print("expected calibration error:", expected_calibration_error(prob, gt_labels))
    print("balanced ece:", expected_calibration_error(prob, gt_labels,
                                                      bins=5))  # print("balanced ece:", expected_calibration_error(prob, gt_labels, bins=2))#

    # gt_labels = np.array([1., 0.])
    # prob = 1. - gt_labels
    # print("Entropy:", entropy(prob))
    # print("negative log likelihood:", nll(prob, gt_labels))
    # print("brier score:", brier_score(prob, gt_labels))
    # print("expected calibration error:", expected_calibration_error(prob, gt_labels))
    return 0


if __name__ == '__main__':
    _main()
