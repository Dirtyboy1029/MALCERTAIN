import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




feature_type = 'drebin'
train_data_type = 'base'

test_data_types = ['earliest', '2013', '2016', 'latest']
MODEL_TYPES = {
    'VBI': 'bayesian',
    'MCD': 'mc_dropout',
    'Ens': 'ensemble',
    'W.Ens': 'wensemble',
    'Ep.Ens':'epoch_ensemble'
}
model_types = ['VBI', 'MCD', 'Ep.Ens', 'Ens', 'W.Ens']
fig, axs = plt.subplots(3, 4, figsize=(24, 12))  # 3 rows for 3 metrics, 4 columns for 4 datasets
metrics = ['Entropy', 'KL-Divergence', 'Standard Deviation']  # 三个指标

for metric_idx, metric_name in enumerate(metrics):  # 遍历每个指标
    for dataset_idx, test_data_type in enumerate(test_data_types):  # 遍历每个数据集
        save_path = '../uncertain_feature/uc_metrics_' + feature_type + '_' + train_data_type + '_' + test_data_type + '.csv'
        data = pd.read_csv(save_path)
        print('load filename and label from ' + save_path)
        gt_labels = np.array(data['gt_labels'].tolist())
        vanilla_prob = np.array(data['vanilla+prob'].tolist())
        pred_labels = (vanilla_prob >= 0.5).astype(np.int32)

        incorrect_sample = list(np.where((gt_labels == 1) & (pred_labels == 0))[0]) + list(
            np.where((gt_labels == 0) & (pred_labels == 1))[0])

        correct_sample = list(np.where((gt_labels == 1) & (pred_labels == 1))[0]) + list(
            np.where((gt_labels == 0) & (pred_labels == 0))[0])

        all_data = []
        for model_type in model_types:
            if metric_name == 'Entropy':
                columns_name = MODEL_TYPES[model_type] + '+entropy'
                metric_values = np.array(data[columns_name].tolist())
            elif metric_name == 'KL-Divergence':
                columns_name = MODEL_TYPES[model_type] + '+kld'
                metric_values = np.array(data[columns_name].tolist())
            elif metric_name == 'Standard Deviation':
                columns_name = MODEL_TYPES[model_type] + '+std'
                metric_values = np.array(data[columns_name].tolist())
            else:
                metric_values = np.zeros(10)

            correct_values = metric_values[correct_sample]
            incorrect_values = metric_values[incorrect_sample]

            all_data.extend([
                ('Correct', model_type, correct_values),
                ('Incorrect', model_type, incorrect_values)
            ])

        # Prepare data for violin plot
        data_flat = []
        for group, model, values in all_data:
            for v in values:
                data_flat.append((group, model, v))

        # Convert data to DataFrame
        df = pd.DataFrame(data_flat, columns=['Group', 'Model', 'Value'])

        ax = axs[metric_idx, dataset_idx]  # 对应的子图
        sns.violinplot(
            data=df,
            x='Model',
            y='Value',
            hue='Group',
            scale='width',
            split=True,
            inner='quartile',
            ax=ax,
            palette={'Correct': 'skyblue', 'Incorrect': 'lightcoral'}
        )
        if metric_idx == 0:
            ax.set_title(test_data_type.capitalize(), fontsize=25, fontweight='bold')
        ax.set_ylim(0, )
        ax.set_xlabel("")
        ax.get_legend().remove()
        if metric_idx == 2 and dataset_idx == 3:
            palette = {'Correct': 'skyblue', 'Incorrect': 'lightcoral'}
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10)
                for key, value in palette.items()
            ]

            ax.legend(handles=handles, loc='upper right', prop={'weight': 'bold', 'size': 15})
        if dataset_idx == 0:
            ax.set_ylabel(metric_name, fontsize=25, fontweight='bold', labelpad=20)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis='both', labelsize=12, width=10)
        if metric_idx != 2:
            ax.set_xticklabels([])
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(20)

        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('metrics_violin.pdf')
plt.show()
