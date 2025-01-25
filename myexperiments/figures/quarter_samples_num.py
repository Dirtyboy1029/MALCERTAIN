# -*- coding: utf-8 -*- 
# @Time : 2024/12/10 10:37 
# @Author : DirtyBoy 
# @File : quarter_samples_num.py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
df = pd.read_csv('../../Datasets/database/malware/malware_drebin.csv')
# 设置中文字体，指定为你电脑中已安装的字体（这里使用SimHei为例）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者使用其他支持中文的字体，如 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示成方块
# 将'time'列转换为日期格式
df['time'] = pd.to_datetime(df['time'])


# 时间归类调整
def adjust_time(row):
    if row['time'].year < 2008:
        return pd.Timestamp('2008-03-31')  # 归类为2009年第一季度
    elif row['time'].year > 2021:
        return pd.Timestamp('2021-12-31')  # 归类为2021年第四季度
    else:
        return row['time']


df['adjusted_time'] = df.apply(adjust_time, axis=1)

# 提取年份和季度
df['year'] = df['adjusted_time'].dt.year
df['quarter'] = df['adjusted_time'].dt.to_period('Y')

# 统计每年每季度的恶意样本数量
quarter_counts = df.groupby(['year', 'quarter']).size().reset_index(name='sample_count')
df = quarter_counts
# 绘制直方图
plt.figure(figsize=(15, 6))
bars = plt.bar(df['year'], df['sample_count'], color='skyblue')
plt.xlabel('年份', fontsize=18, fontweight='bold')
plt.ylabel('恶意样本数量', fontsize=18, fontweight='bold')
plt.title('每年恶意样本数量', fontsize=22, fontweight='bold')
plt.xticks(df['year'],)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=14, fontweight='bold')  # 设置x轴刻度字体加粗加大
plt.yticks(fontsize=14, fontweight='bold')  # 设置y轴刻度字体加粗加大
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, str(int(yval)), ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('1.png')
# 显示图形
plt.show()
