# -*- coding: utf-8 -*- 
# @Time : 2023/11/23 12:53 
# @Author : DirtyBoy 
# @File : feature_extractor.py
from core.feature.feature_extraction import DrebinFeature, OpcodeSeq, MultiModality, APISequence
import os

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p',  type=str, default='/home/lhd/apk')
    parser.add_argument('-feature_type', '-ft', type=str, default='opcode')
    args = parser.parse_args()
    p = args.p
    feature_type = args.feature_type

    if feature_type == 'drebin':
        Feature = DrebinFeature
    elif feature_type == 'opcode':
        Feature = OpcodeSeq
    elif feature_type == 'apiseq':
        Feature = APISequence
    elif feature_type == 'multimodality':
        Feature = MultiModality
    else:
        Feature = None

    android_features_saving_dir = '/home/lhd/MalCertain/Training/feture_file/' + feature_type
    intermediate_data_saving_dir = '/home/lhd/MalCertain/feture_file/' + feature_type

    feature_extractor = Feature(android_features_saving_dir, intermediate_data_saving_dir, update=False,
                                proc_number=12)

    feature_extractor.feature_extraction(p)
