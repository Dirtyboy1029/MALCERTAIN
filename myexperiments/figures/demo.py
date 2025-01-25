# -*- coding: utf-8 -*- 
# @Time : 2024/12/8 21:59 
# @Author : DirtyBoy 
# @File : demo.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score
test_data_types = ['earliest', '2013', '2016', 'latest']
MODEL_TYPES = {
    'Bay.': 'bayesian',
    'MC': 'mc_dropout',
    'Ens': 'ensemble',
    'W.ens': 'wensemble',
    'Ep.ens': 'epoch_ensemble'
}
METRICS_TYPES = {'Entropy': 'entropy',
                 'KL-Divergence': 'kld',
                 'Standard Deviation': 'std'}
model_types = ['Bay.',  'Ep.ens', 'MC','Ens', 'W.ens']
metrics_types = ['Entropy', 'KL-Divergence', 'Standard Deviation']


def polt_roc(fig, feature_type, train_data_type, test_data_type, model_type, metrics_type, color):
    save_path = '../uncertain_feature/uc_metrics_' + feature_type + '_' + train_data_type + '_' + test_data_type + '.csv'
    data = pd.read_csv(save_path)
    gt_labels = np.array(data['gt_labels'].tolist())
    vanilla_prob = np.array(data['vanilla+prob'].tolist())
    pred_labels = (vanilla_prob >= 0.5).astype(np.int32)

    correct_mask = (gt_labels != pred_labels).astype(int)

    uc_metrics_name = model_type + '+' + metrics_type
    uc_metrics = np.array(data[uc_metrics_name].tolist())
    fpr, tpr, _ = roc_curve(correct_mask, uc_metrics)
    roc_auc = auc(fpr, tpr)
    fig.plot(fpr, tpr, label=f'{uc_metrics_name} (AUC={roc_auc:.3f})', color=color, lw=2)


# 创建子图
plt.style.use('default')
fig, axes = plt.subplots(3, 5, figsize=(25, 12))
colors = ['red', 'blue', 'green', 'purple']

# 绘制每个指标的子图
for metrics_ids, metrics in enumerate(metrics_types):
    for model_ids, model_type in enumerate(model_types):  # 每个指标有4组
        ax = axes[metrics_ids, model_ids]
        for i, test_data in enumerate(test_data_types):
            polt_roc(ax, feature_type='drebin', train_data_type='base', test_data_type=test_data,
                     model_type=MODEL_TYPES[model_type], metrics_type=METRICS_TYPES[metrics], color=colors[i])
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1)  # 随机猜测线
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        # ax.set_title(f'Metric {i + 1}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
