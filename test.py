# -*- coding: utf-8 -*-
"""
分析各个模型预测错误样本（all_AIS_correct=0）的特征和标签分布
1. 绘制连续特征、离散特征、损伤标量、AIS/MAIS的一维概率密度分布对比图
   - 模型内部：正确 vs 错误
   - 跨模型：各模型的错误样本对比
2. 将错误样本的case_id整理到Excel的不同sheet，并标记是否在Injury_labels中
   - 区分训练集和验证+测试集的统计
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# --- 配置区 ---
CSV_FILES = [
    r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\full_dataset_predictions_2005354_20252025.csv",
    r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\full_dataset_predictions_2005354_123.csv",
    r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\full_dataset_predictions_2752354_20252025.csv",
    r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\full_dataset_predictions_2752354_123.csv",
    r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\full_dataset_predictions_2640466_20252025.csv",
    r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\full_dataset_predictions_2640466_123.csv"
]

INJURY_LABELS_FILE = r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\Injury_labels_1023.xlsx"

BASE_OUTPUT_DIR = r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关"
OUTPUT_SUBDIR_NAME = "incorrect_predictions_analysis"

# 定义特征列名（与 dataset_prepare.py 一致）
CONTINUOUS_FEATURES = [
    'impact_velocity', 'impact_angle', 'overlap', 'll1', 'll2', 'btf', 
    'pp', 'plp', 'llattf', 'ptf', 'aft', 'ttf', 'sp', 'recline_angle'
]

DISCRETE_FEATURES = [
    'occupant_type', 'lla_status', 'dz', 'aav_status'
]

INJURY_SCALARS = ['HIC_true', 'Dmax_true', 'Nij_true']
AIS_LABELS = ['AIS_head_true_raw', 'AIS_chest_true_raw', 'AIS_neck_true_raw', 'MAIS_true_raw']

# --- 结束配置 ---


def ensure_output_directory():
    """确保输出子文件夹存在"""
    output_dir = os.path.join(BASE_OUTPUT_DIR, OUTPUT_SUBDIR_NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ 创建输出目录: {output_dir}")
    return output_dir


def load_and_split_data(csv_file):
    """加载数据并分为正确/错误两组，同时区分训练集和验证+测试集"""
    df = pd.read_csv(csv_file)
    df = df[df['dataset_type'] != 'unassigned'].copy()
    
    # 分为训练集和验证+测试集
    train_df = df[df['dataset_type'] == 'train'].copy()
    val_test_df = df[df['dataset_type'].isin(['valid', 'test'])].copy()
    
    # 进一步划分正确/错误
    train_correct = train_df[train_df['all_AIS_correct'] == 1].copy()
    train_incorrect = train_df[train_df['all_AIS_correct'] == 0].copy()
    val_test_correct = val_test_df[val_test_df['all_AIS_correct'] == 1].copy()
    val_test_incorrect = val_test_df[val_test_df['all_AIS_correct'] == 0].copy()
    
    model_name = os.path.basename(csv_file).replace('full_dataset_predictions_', '').replace('.csv', '')
    
    return {
        'train_correct': train_correct,
        'train_incorrect': train_incorrect,
        'val_test_correct': val_test_correct,
        'val_test_incorrect': val_test_incorrect,
        'model_name': model_name
    }


def plot_continuous_distribution(data_dict, output_dir):
    """绘制连续特征的KDE分布对比（仅KDE曲线）"""
    model_name = data_dict['model_name']
    n_features = len(CONTINUOUS_FEATURES)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(CONTINUOUS_FEATURES):
        ax = axes[i]
        
        # 训练+验证+测试集的正确和错误样本合并
        correct_data = pd.concat([data_dict['train_correct'][feature], data_dict['val_test_correct'][feature]]).dropna()
        incorrect_data = pd.concat([data_dict['train_incorrect'][feature], data_dict['val_test_incorrect'][feature]]).dropna()

        # 仅绘制KDE曲线
        if len(correct_data) > 1:
            correct_data.plot.kde(ax=ax, color='green', linewidth=2.5, label='Correct (all_dataset)')
        
        if len(incorrect_data) > 1:
            incorrect_data.plot.kde(ax=ax, color='red', linewidth=2.5, linestyle='--', label='Incorrect (all_dataset)')
        
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余子图
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f'Model {model_name} - Continuous Features KDE (Correct vs Incorrect, all_dataset)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_continuous_kde.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 连续特征KDE分布图已保存")


def plot_discrete_distribution(data_dict, output_dir):
    """绘制离散特征的概率分布对比（归一化条形图）"""
    model_name = data_dict['model_name']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(DISCRETE_FEATURES):
        ax = axes[i]
        
        # 训练+验证+测试集的正确和错误样本合并
        correct_counts = pd.concat([data_dict['train_correct'][feature], data_dict['val_test_correct'][feature]]).value_counts(normalize=True).sort_index() * 100
        incorrect_counts = pd.concat([data_dict['train_incorrect'][feature], data_dict['val_test_incorrect'][feature]]).value_counts(normalize=True).sort_index() * 100
        
        all_categories = sorted(set(correct_counts.index) | set(incorrect_counts.index))
        correct_counts = correct_counts.reindex(all_categories, fill_value=0)
        incorrect_counts = incorrect_counts.reindex(all_categories, fill_value=0)
        
        x = np.arange(len(all_categories))
        width = 0.35
        
        ax.bar(x - width/2, correct_counts, width, label='Correct (all_dataset)', 
               color='lightgreen', alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, incorrect_counts, width, label='Incorrect (all_dataset)', 
               color='lightcoral', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Probability (%)', fontsize=12)
        ax.set_title(f'{feature}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Model {model_name} - Discrete Features Distribution (Correct vs Incorrect, all_dataset)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_discrete_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 离散特征分布图已保存")


def plot_injury_scalars_distribution(data_dict, output_dir):
    """绘制损伤标量的KDE分布对比"""
    model_name = data_dict['model_name']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, injury in enumerate(INJURY_SCALARS):
        ax = axes[i]
        
        # 训练+验证+测试集的正确和错误样本合并
        correct_data = pd.concat([data_dict['train_correct'][injury], data_dict['val_test_correct'][injury]]).dropna()
        incorrect_data = pd.concat([data_dict['train_incorrect'][injury], data_dict['val_test_incorrect'][injury]]).dropna()
        
        if len(correct_data) > 1:
            correct_data.plot.kde(ax=ax, color='green', linewidth=2.5, label='Correct (all_dataset)')
        
        if len(incorrect_data) > 1:
            incorrect_data.plot.kde(ax=ax, color='red', linewidth=2.5, linestyle='--', label='Incorrect (all_dataset)')
        
        ax.set_title(f'{injury.replace("_true", "")}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Model {model_name} - Injury Scalars KDE (Correct vs Incorrect, all_dataset)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_injury_scalars_kde.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 损伤标量KDE分布图已保存")


def plot_ais_mais_distribution(data_dict, output_dir):
    """绘制AIS和MAIS等级的概率分布对比"""
    model_name = data_dict['model_name']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, label in enumerate(AIS_LABELS):
        ax = axes[i]
        
        # 训练+验证+测试集的正确和错误样本合并
        correct_counts = pd.concat([data_dict['train_correct'][label], data_dict['val_test_correct'][label]]).value_counts(normalize=True).sort_index() * 100
        incorrect_counts = pd.concat([data_dict['train_incorrect'][label], data_dict['val_test_incorrect'][label]]).value_counts(normalize=True).sort_index() * 100

        all_levels = sorted(set(correct_counts.index) | set(incorrect_counts.index))
        correct_counts = correct_counts.reindex(all_levels, fill_value=0)
        incorrect_counts = incorrect_counts.reindex(all_levels, fill_value=0)
        
        x = np.arange(len(all_levels))
        width = 0.35
        
        ax.bar(x - width/2, correct_counts, width, label='Correct (all_dataset)', 
               color='lightgreen', alpha=0.8, edgecolor='black')
        ax.bar(x + width/2, incorrect_counts, width, label='Incorrect (all_dataset)', 
               color='lightcoral', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('AIS/MAIS Level', fontsize=12)
        ax.set_ylabel('Probability (%)', fontsize=12)
        ax.set_title(f'{label.replace("_true_raw", "")}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_levels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Model {model_name} - AIS/MAIS Distribution (Correct vs Incorrect, all_dataset)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_ais_mais_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ AIS/MAIS分布图已保存")


def plot_cross_model_comparison(all_data_dicts, output_dir):
    """跨模型对比：对比各模型的错误样本（包含训练集+验证+测试集）"""
    print("\n  【跨模型对比】对比各模型的错误样本（全部数据集）...")
    
    colors = ['red', 'blue', 'purple', 'orange', 'brown']
    linestyles = ['-', '--', '-.', ':', '-']
    
    # 1. 连续特征跨模型对比
    n_features = len(CONTINUOUS_FEATURES)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(CONTINUOUS_FEATURES):
        ax = axes[i]
        
        for j, data_dict in enumerate(all_data_dicts):
            # 合并训练集和验证+测试集的错误样本
            train_incorrect_data = data_dict['train_incorrect'][feature].dropna()
            val_test_incorrect_data = data_dict['val_test_incorrect'][feature].dropna()
            all_incorrect_data = pd.concat([train_incorrect_data, val_test_incorrect_data])
            
            if len(all_incorrect_data) > 1:
                all_incorrect_data.plot.kde(
                    ax=ax, 
                    color=colors[j % len(colors)], 
                    linewidth=2, 
                    linestyle=linestyles[j % len(linestyles)],
                    label=f"{data_dict['model_name']}"
                )
        
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Cross-Model Comparison - Continuous Features (Incorrect Samples, All Datasets)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_model_continuous_kde.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 跨模型连续特征对比图已保存")
    
    # 2. 损伤标量跨模型对比 - 同样修改
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, injury in enumerate(INJURY_SCALARS):
        ax = axes[i]
        
        for j, data_dict in enumerate(all_data_dicts):
            train_incorrect_data = data_dict['train_incorrect'][injury].dropna()
            val_test_incorrect_data = data_dict['val_test_incorrect'][injury].dropna()
            all_incorrect_data = pd.concat([train_incorrect_data, val_test_incorrect_data])
            
            if len(all_incorrect_data) > 1:
                all_incorrect_data.plot.kde(
                    ax=ax, 
                    color=colors[j % len(colors)], 
                    linewidth=2.5, 
                    linestyle=linestyles[j % len(linestyles)],
                    label=f"{data_dict['model_name']}"
                )
        
        ax.set_title(f'{injury.replace("_true", "")}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Cross-Model Comparison - Injury Scalars (Incorrect Samples, All Datasets)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_model_injury_scalars_kde.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 跨模型损伤标量对比图已保存")
    
    # 3. AIS/MAIS跨模型对比 - 同样修改
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, label in enumerate(AIS_LABELS):
        ax = axes[i]
        
        # 收集所有模型的类别
        all_levels = set()
        for data_dict in all_data_dicts:
            all_levels.update(data_dict['train_incorrect'][label].unique())
            all_levels.update(data_dict['val_test_incorrect'][label].unique())
        all_levels = sorted(all_levels)
        
        x = np.arange(len(all_levels))
        width = 0.8 / len(all_data_dicts)
        
        for j, data_dict in enumerate(all_data_dicts):
            # 合并训练集和验证+测试集
            all_incorrect = pd.concat([
                data_dict['train_incorrect'][label],
                data_dict['val_test_incorrect'][label]
            ])
            counts = all_incorrect.value_counts(normalize=True).sort_index() * 100
            counts = counts.reindex(all_levels, fill_value=0)
            
            ax.bar(x + j*width - 0.4 + width/2, counts, width, 
                   label=data_dict['model_name'],
                   alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('AIS/MAIS Level', fontsize=12)
        ax.set_ylabel('Probability (%)', fontsize=12)
        ax.set_title(f'{label.replace("_true_raw", "")}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_levels)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Cross-Model Comparison - AIS/MAIS Distribution (Incorrect Samples, All Datasets)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_model_ais_mais.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 跨模型AIS/MAIS对比图已保存")


def create_case_id_summary(csv_files, injury_labels_file, output_dir):
    """创建包含所有模型错误预测case_id的汇总Excel（任务2，区分训练集和验证+测试集）"""
    # 加载目标case_id集合
    injury_df = pd.read_excel(injury_labels_file)
    target_case_ids = set(injury_df['case_id'].values)
    print(f"\n【任务2】生成错误预测case_id汇总表")
    print(f"  - 已加载Injury_labels，共 {len(target_case_ids)} 个case_id")
    
    output_file = os.path.join(output_dir, "incorrect_case_ids_summary.xlsx")
    
    # 用于收集每个模型的错误case_id集合
    model_incorrect_sets = {}
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 创建汇总统计sheet
        summary_stats = []
        
        for csv_file in csv_files:
            model_name = os.path.basename(csv_file).replace('full_dataset_predictions_', '').replace('.csv', '')
            
            # 读取数据
            df = pd.read_csv(csv_file)
            df = df[df['dataset_type'] != 'unassigned'].copy()
            
            # 分别处理训练集和验证+测试集
            train_df = df[df['dataset_type'] == 'train']
            val_test_df = df[df['dataset_type'].isin(['valid', 'test'])]
            
            train_incorrect = train_df[train_df['all_AIS_correct'] == 0]
            val_test_incorrect = val_test_df[val_test_df['all_AIS_correct'] == 0]
            
            # 训练集case_id
            train_case_ids = sorted(train_incorrect['case_id'].unique())
            train_summary = pd.DataFrame({
                'case_id': train_case_ids,
                'dataset_split': ['train'] * len(train_case_ids),
                'in_Injury_labels': [1 if cid in target_case_ids else 0 for cid in train_case_ids]
            })
            
            # 验证+测试集case_id
            val_test_case_ids = sorted(val_test_incorrect['case_id'].unique())
            val_test_summary = pd.DataFrame({
                'case_id': val_test_case_ids,
                'dataset_split': ['val_test'] * len(val_test_case_ids),
                'in_Injury_labels': [1 if cid in target_case_ids else 0 for cid in val_test_case_ids]
            })
            
            # 合并
            combined_summary = pd.concat([train_summary, val_test_summary], ignore_index=True)
            combined_summary.to_excel(writer, sheet_name=model_name, index=False)
            
            # 收集该模型的所有错误case_id（用于后续交集计算）
            all_incorrect_case_ids = set(train_case_ids) | set(val_test_case_ids)
            model_incorrect_sets[model_name] = all_incorrect_case_ids
            
            # 统计信息
            train_in_target = train_summary['in_Injury_labels'].sum()
            val_test_in_target = val_test_summary['in_Injury_labels'].sum()
            
            summary_stats.append({
                'Model': model_name,
                'Train_Incorrect_Count': len(train_case_ids),
                'Train_In_InjuryLabels': train_in_target,
                'Train_Error_Rate': f"{len(train_incorrect)/len(train_df)*100:.2f}%" if len(train_df) > 0 else "N/A",
                'ValTest_Incorrect_Count': len(val_test_case_ids),
                'ValTest_In_InjuryLabels': val_test_in_target,
                'ValTest_Error_Rate': f"{len(val_test_incorrect)/len(val_test_df)*100:.2f}%" if len(val_test_df) > 0 else "N/A"
            })
            
            print(f"  - Model '{model_name}':")
            print(f"      训练集: {len(train_case_ids)} 个错误case_id, 其中 {train_in_target} 个在Injury_labels中")
            print(f"      验证+测试集: {len(val_test_case_ids)} 个错误case_id, 其中 {val_test_in_target} 个在Injury_labels中")
        
        # 写入汇总统计sheet
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # # === 新增：三个模型共有的错误样本 ===
        # if len(model_incorrect_sets) == 3:
        #     model_names = list(model_incorrect_sets.keys())
        #     common_all_three = model_incorrect_sets[model_names[0]] & \
        #                        model_incorrect_sets[model_names[1]] & \
        #                        model_incorrect_sets[model_names[2]]
            
        #     common_all_three_sorted = sorted(common_all_three)
        #     common_all_three_df = pd.DataFrame({
        #         'case_id': common_all_three_sorted,
        #         'in_Injury_labels': [1 if cid in target_case_ids else 0 for cid in common_all_three_sorted]
        #     })
        #     common_all_three_df.to_excel(writer, sheet_name='Common_All_3_Models', index=False)
            
        #     in_injury_count = common_all_three_df['in_Injury_labels'].sum()
        #     print(f"\n  【三模型共有错误样本】")
        #     print(f"      共有 {len(common_all_three_sorted)} 个case_id")
        #     print(f"      其中 {in_injury_count} 个在Injury_labels中 ({in_injury_count/len(common_all_three_sorted)*100:.1f}%)")
        
        # === 新增：共有的错误样本 ===
        model_1 = None
        model_2 = None
        
        for model_name, case_set in model_incorrect_sets.items():
            if '2640466_20252025' in model_name:
                model_1 = case_set
            elif '2640466_123' in model_name:
                model_2 = case_set
        
        if model_1 is not None and model_2 is not None:
            common_two = model_1 & model_2
            common_two_sorted = sorted(common_two)
            common_two_df = pd.DataFrame({
                'case_id': common_two_sorted,
                'in_Injury_labels': [1 if cid in target_case_ids else 0 for cid in common_two_sorted]
            })
            common_two_df.to_excel(writer, sheet_name='Common_2640466_123_2640466_20252025', index=False)
            
            in_injury_count_two = common_two_df['in_Injury_labels'].sum()
            print(f"\n  【2640466_123和2640466_20252025共有错误样本】")
            print(f"      共有 {len(common_two_sorted)} 个case_id")
            print(f"      其中 {in_injury_count_two} 个在Injury_labels中 ({in_injury_count_two/len(common_two_sorted)*100:.1f}%)")
    
    print(f"\n  ✓ Case ID汇总表已保存: {output_file}\n")

def main():
    print("="*70)
    print("开始分析预测错误样本（all_AIS_correct=0）的分布")
    print("="*70)
    
    # 确保输出目录存在
    output_dir = ensure_output_directory()
    
    # 加载所有模型的数据
    all_data_dicts = []
    
    # 任务1: 为每个模型生成分布对比图
    print("\n【任务1】生成分布对比图")
    for csv_file in CSV_FILES:
        print(f"\n处理文件: {os.path.basename(csv_file)}")
        
        data_dict = load_and_split_data(csv_file)
        all_data_dicts.append(data_dict)
        
        # 打印详细统计（区分训练集和验证+测试集）
        train_total = len(data_dict['train_correct']) + len(data_dict['train_incorrect'])
        val_test_total = len(data_dict['val_test_correct']) + len(data_dict['val_test_incorrect'])
        
        print(f"  【训练集】")
        print(f"    - 正确预测: {len(data_dict['train_correct'])}, 错误预测: {len(data_dict['train_incorrect'])}")
        if train_total > 0:
            print(f"    - 错误率: {len(data_dict['train_incorrect'])/train_total*100:.2f}%")
        
        print(f"  【验证+测试集】")
        print(f"    - 正确预测: {len(data_dict['val_test_correct'])}, 错误预测: {len(data_dict['val_test_incorrect'])}")
        if val_test_total > 0:
            print(f"    - 错误率: {len(data_dict['val_test_incorrect'])/val_test_total*100:.2f}%")
        
        # 生成单模型对比图（基于验证+测试集）
        plot_continuous_distribution(data_dict, output_dir)
        plot_discrete_distribution(data_dict, output_dir)
        plot_injury_scalars_distribution(data_dict, output_dir)
        plot_ais_mais_distribution(data_dict, output_dir)
    
    # 新增：跨模型对比（仅错误样本）
    plot_cross_model_comparison(all_data_dicts, output_dir)
    
    # 任务2: 生成case_id汇总表
    create_case_id_summary(CSV_FILES, INJURY_LABELS_FILE, output_dir)
    
    print("="*70)
    print("所有分析完成！")
    print(f"结果已保存至: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()