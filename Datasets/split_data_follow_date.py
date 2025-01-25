# -*- coding: utf-8 -*- 
# @Time : 2024/9/19 15:20 
# @Author : DirtyBoy 
# @File : split_data_follow_date.py
import pandas as pd

data_type = 'malware'
feature_type = 'drebin'
df = pd.read_csv('database/' + data_type + '/' + data_type + '_' + feature_type + '.csv')

# 将'出现时间'列转换为datetime格式
df['time'] = pd.to_datetime(df['time'])

# 划分节点列表（时间点）
split_points_list = ['2012-06-01', '2012-08-01', '2013-01-01', '2016-01-01', '2023-01-01', ]

# 将split_points转换为datetime格式，并增加无穷大范围
split_points = pd.to_datetime(split_points_list)
split_points = [pd.Timestamp.min] + list(split_points) + [pd.Timestamp.max]

# 使用pd.cut进行划分
df['datasets'] = pd.cut(df['time'], bins=split_points)

# 按区间显示数据
for i, (interval, group) in enumerate(df.groupby('datasets')):
    if i < len(split_points_list):
        print(f"区间 {interval}:")
        print(group.shape)
        c1, c2, c3 = split_points_list[i].split('-')
        group.to_csv('database/' + data_type + '/' + data_type + '_' + c1 + c2 + c3 + '.csv')
print(df.shape)

# from Datasets.utils import save_to_txt
# import pandas as pd
#
# df = pd.read_csv('androzoo_download/2012_Benign.csv')
#
# # 将'时间戳'列转换为datetime格式
# df['dex_date1'] = pd.to_datetime(df['dex_date'])
#
# # 指定分割的日期为2012年6月1日
# cutoff_date1 = pd.Timestamp('2012-06-01')
# cutoff_date2 = pd.Timestamp('2012-08-01')
# # 按照时间戳将数据分为两组
# before_cutoff = df[df['dex_date1'] < cutoff_date2]
# before_cutoff = before_cutoff[before_cutoff['dex_date1'] > cutoff_date1]
# a = before_cutoff['sha256'].tolist()
# a = [item.lower() for item in a]
# save_to_txt(a, 'androzoo_download/tmp.txt')
