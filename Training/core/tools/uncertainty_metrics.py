# -*- coding: utf-8 -*- 
# @Time : 2023/4/17 16:15 
# @Author : DirtyBoy 
# @File : uncertainty_metrics.py
import numpy as np


def gauss_to_pi(y_pred_gauss_mid_all, y_pred_gauss_dev_all, n_std_devs):
    """
    input is individual NN estimates of mean and std dev
    1. combine into ensemble estimates of mean and std dev
    2. convert to prediction intervals
    """
    # 1. merge to one estimate (described in paper, mixture of gaussians)
    y_pred_gauss_mid = np.mean(y_pred_gauss_mid_all, axis=0)
    y_pred_gauss_dev = np.sqrt(np.mean(np.square(y_pred_gauss_dev_all) \
                                       + np.square(y_pred_gauss_mid_all), axis=0) - np.square(y_pred_gauss_mid))

    # 2. create pi's
    y_pred_U = y_pred_gauss_mid + n_std_devs * y_pred_gauss_dev
    y_pred_L = y_pred_gauss_mid - n_std_devs * y_pred_gauss_dev

    return y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, y_pred_L


def pi_to_gauss(y_pred_all, lube_perc, perc_or_norm, n_std_devs):
    """
    input is individual NN estimates of upper and lower bounds
    1. combine into ensemble estimates of upper and lower bounds
    2. convert to mean and std dev of gaussian
    y_pred_all is shape [no. ensemble, no. predictions, 2]
    """
    in_ddof = 1 if y_pred_all.shape[0] > 1 else 0

    lube_perc_U = lube_perc
    lube_perc_L = 100. - lube_perc
    if perc_or_norm == 'perc':
        y_pred_U = np.percentile(y_pred_all[:, :, 0], q=lube_perc_U, axis=0, interpolation='linear')
        y_pred_L = np.percentile(y_pred_all[:, :, 1], q=lube_perc_L, axis=0, interpolation='linear')
    elif perc_or_norm == 'norm':
        # this is std err of mean STEM
        y_pred_U = np.mean(y_pred_all[:, :, 0], axis=0) + 1.96 * np.std(y_pred_all[:, :, 0], axis=0,
                                                                        ddof=in_ddof) / np.sqrt(y_pred_all.shape[0])
        y_pred_L = np.mean(y_pred_all[:, :, 1], axis=0) - 1.96 * np.std(y_pred_all[:, :, 1], axis=0,
                                                                        ddof=in_ddof) / np.sqrt(y_pred_all.shape[0])
    # actually we used the above in experiments, but now believe the below is better justified theoretically
    # it's not vitally important as can account for this using lambda to encourage tighter PIs to balance out
    # e.g. if lambda was reported as 15., reduce to ~ 4. when using the below

    # not STEM, just model uncert
    # y_pred_U = np.mean(y_pred_all[:,:,0],axis=0) + 1.96*np.std(y_pred_all[:,:,0],axis=0, ddof=in_ddof)
    # y_pred_L = np.mean(y_pred_all[:,:,1],axis=0) - 1.96*np.std(y_pred_all[:,:,1],axis=0, ddof=in_ddof)

    # need to do this before calc mid and std dev
    y_pred_U_temp = np.maximum(y_pred_U, y_pred_L)
    y_pred_L = np.minimum(y_pred_U, y_pred_L)
    y_pred_U = y_pred_U_temp

    y_pred_gauss_mid = np.mean((y_pred_U, y_pred_L), axis=0)
    y_pred_gauss_dev = (y_pred_U - y_pred_gauss_mid) / n_std_devs

    return y_pred_gauss_mid, y_pred_gauss_dev, y_pred_U, y_pred_L


def picp(y_true, y_lower, y_upper):
    """
    Prediction Interval Coverage Probability (PICP). Computes the fraction of samples for which the grounds truth lies
    within predicted interval. Measures the prediction interval calibration for regression.
    Args:
        y_true: Ground truth
        y_lower: predicted lower bound
        y_upper: predicted upper bound
    Returns:
        float: the fraction of samples for which the grounds truth lies within predicted interval.
    """
    satisfies_upper_bound = y_true <= y_upper
    satisfies_lower_bound = y_true >= y_lower
    return np.mean(satisfies_upper_bound * satisfies_lower_bound)


def mpiw(y_lower, y_upper):
    """
    Mean Prediction Interval Width (MPIW). Computes the average width of the the prediction intervals. Measures the
    sharpness of intervals.
    Args:
        y_lower: predicted lower bound
        y_upper: predicted upper bound
    Returns:
        float: the average width the prediction interval across samples.
    """
    return np.mean(np.abs(y_lower - y_upper))


if __name__ == '__main__':
    alpha = 0.05
    if alpha == 0.05:
        n_std_devs = 1.96
    elif alpha == 0.10:
        n_std_devs = 1.645
    elif alpha == 0.01:
        n_std_devs = 2.575
    else:
        raise Exception('ERROR unusual alpha')
    y_pred_gauss_mid_all = [-1.6406342, -0.7374642, 0.45904014, -0.44731274, 3.6834389e-02]
    y_pred_gauss_dev_all = [-82.20176, -114.08026, -85.39287, -77.25536, -4.2187904e+01]
    y_pred_gauss_mid = np.mean(y_pred_gauss_mid_all, axis=0)
    y_pred_gauss_dev = np.sqrt(np.mean(np.square(y_pred_gauss_dev_all) \
                                       + np.square(y_pred_gauss_mid_all), axis=0) - np.square(y_pred_gauss_mid))

    y_pred_U = y_pred_gauss_mid + n_std_devs * y_pred_gauss_dev
    y_pred_L = y_pred_gauss_mid - n_std_devs * y_pred_gauss_dev
    print(y_pred_U,y_pred_L)
    print(np.array([0.23219389,-0.3638082,-2.2652895,-1.4566234,-0.86558557])<=np.array(y_pred_gauss_dev_all))
    print(picp(np.array([0.23219389,-0.3638082,-2.2652895,-1.4566234,-0.86558557]),np.array(y_pred_gauss_dev_all),np.array(y_pred_gauss_mid_all)))
