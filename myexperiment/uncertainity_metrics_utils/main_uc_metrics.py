# -*- coding: utf-8 -*- 

# @File : main_uc_metrics.py
from experiments.oos import data_preprocessing_get_name as oos_data_preprocessing
from experiments.adv import data_preprocessing, data_preprocessing_get_name
from myexperiment.uncertainity_metrics_utils.tools import Save_to_Csv
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_arch', type=str, default="drebin")
    parser.add_argument('-model_type', type=str, default="small")
    parser.add_argument('-data_type', type=str, default="adv")
    args = parser.parse_args()
    model_type = args.model_type
    if args.model_arch == "multi":
        architecture_type = "multimodalitynn"
        feature_type = "multimodality"
        model_dir = "multi_model/"
    elif args.model_arch == "drebin":
        architecture_type = "dnn"
        feature_type = "drebin"
        model_dir = "drebin_model/"
    base_model_path = "/home/lhd/uncertainity_malware/myexperiment/model/" + model_dir + model_type + "_model/"

    if args.data_type == "adv":
        from myexperiment.uncertainity_metrics_utils.adv_utils import get_label, epoch_ensemble_uc, \
            get_deep_ensemble_uc, \
            get_mc_dropout_uc, get_bay_uc, \
            get_weight_ensemble_uc

        model_path = os.path.join(base_model_path, "vanilla/epoch20")
        prist_data, adv_data, prist_y, adv_y, input_dim, prist_filenames, adv_filenames = data_preprocessing_get_name(
            feature_type, proc_numbers=2)
        get_label(model_path, prist_data, adv_data, prist_filenames, adv_filenames, architecture_type)

        ##轮次集成
        base_path = os.path.join(base_model_path, "vanilla/")
        epoch_ent_pst, epoch_ent_adv, epoch_kld_pst, epoch_kld_adv, epoch_std_pst, epoch_std_adv = epoch_ensemble_uc(
            base_path=base_path, prist_data=prist_data, adv_data=adv_data, prist_filenames=prist_filenames,
            adv_filenames=adv_filenames, true_flase=True, architecture_type=architecture_type)

        ##mc-dropout entropy
        

        get_mc_dropout_uc(prist_data=prist_data, adv_data=adv_data, epoch='30', prist_filenames=prist_filenames,
                          adv_filenames=adv_filenames, architecture_type=architecture_type,
                          model_directory=os.path.join(base_model_path, "mc_dropout/epoch30"), true_flase=True)

        ##bayesian entropy
        

        get_bay_uc(prist_data=prist_data, adv_data=adv_data, epoch='30', prist_filenames=prist_filenames,
                   adv_filenames=adv_filenames, model_directory=os.path.join(base_model_path, "bayesian/epoch20"),
                   architecture_type=architecture_type, true_flase=True)

        ##deep ensemble entropy
        

        get_deep_ensemble_uc(prist_data=prist_data, adv_data=adv_data, epoch=30, n_members=10,
                             prist_filenames=prist_filenames, adv_filenames=adv_filenames,
                             model_directory=os.path.join(base_model_path, "ensemble/n_member10/epoch30"),
                             architecture_type=architecture_type, true_flase=True)

        

        ##weight ensemble entropy
       

        get_weight_ensemble_uc(prist_data=prist_data, adv_data=adv_data,
                               model_directory=os.path.join(base_model_path, "wensemble/n_member10/epoch30"),
                               epoch=30, n_members=10, prist_filenames=prist_filenames, adv_filenames=adv_filenames,
                               architecture_type=architecture_type, true_flase=True)


    else:
        from myexperiment.uncertainity_metrics_utils.utils import get_label, epoch_ensemble_uc, get_deep_ensemble_uc, \
            get_mc_dropout_uc, get_bay_uc, \
            get_weight_ensemble_uc

        model_path = os.path.join(base_model_path, "vanilla/epoch20")
        ood_data, oos_y, input_dim, oos_filenames = oos_data_preprocessing(feature_type, proc_numbers=2,
                                                                           data_type=args.data_type)
        get_label(model_path, ood_data, oos_y, oos_filenames, architecture_type)

        base_path = os.path.join(base_model_path, "vanilla")
        epoch_ensemble_uc(base_path, ood_data, oos_y, oos_filenames, architecture_type)

       
        get_bay_uc(model_directory=os.path.join(base_model_path, "bayesian/epoch30"), ood_data=ood_data, oos_y=oos_y,
                   oos_filenames=oos_filenames, epoch=30, architecture_type=architecture_type)

        get_mc_dropout_uc(model_directory=os.path.join(base_model_path, "mc_dropout/epoch30"), ood_data=ood_data,
                          oos_y=oos_y,
                          oos_filenames=oos_filenames, epoch=30, architecture_type=architecture_type)

       
        get_deep_ensemble_uc(model_directory=os.path.join(base_model_path, "ensemble/n_member10/epoch30"),
                             ood_data=ood_data, architecture_type=architecture_type,
                             oos_y=oos_y,
                             oos_filenames=oos_filenames, epoch=30, n_members=10)

        get_weight_ensemble_uc(model_directory=os.path.join(base_model_path, "wensemble/n_member10/epoch30"),
                               ood_data=ood_data, architecture_type=architecture_type,
                               oos_y=oos_y,
                               oos_filenames=oos_filenames, epoch=30, n_members=10)
