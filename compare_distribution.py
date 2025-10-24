# -*- coding: utf-8 -*-
"""
compare_distributions.py

加载两个 distribution 文件，根据 is_pulse_ok 和 is_injury_ok 
进行过滤，然后计算并比较所有18个标量特征、3个损伤标量
以及4个AIS/MAIS等级的分布。
"""

import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 关键：从您的项目中导入 AIS 计算函数 ---
# 确保此脚本与 utils 文件夹在同一项目结构下
try:
    from AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck
except ImportError:
    print("错误：无法导入 utils.AIS_cal。请确保此脚本在项目根目录下运行。")
    exit()

# --- 1. 配置区 ---

# 1.1) 第一个 distribution 文件（例如，旧数据）
FILE_A_PATH = r"E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1022.csv"
FILE_A_LABEL = "Data_Oct22"  # 用于图例的标签

# 1.2) 第二个 distribution 文件（例如，新数据）
FILE_B_PATH = r"E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1024.csv"
FILE_B_LABEL = "Data_Oct24"  # 用于图例的标签

# 1.3) 保存对比图表的输出目录
OUTPUT_DIR = "./distribution_comparison2"

# --- 结束配置 ---


def load_and_filter_data(filepath, file_label):
    """加载、过滤并标记数据源"""
    print(f"正在加载文件: {filepath} (标记为: {file_label})")
    
    # 定义所有需要分析的列
    # (基于 data_package.py 中的18个特征 和 3个标签)
    required_features = [
        'impact_velocity', 'impact_angle', 'overlap', 'occupant_type',
        'll1', 'll2', 'btf', 'pp', 'plp', 'lla_status', 'llattf',
        'dz', 'ptf', 'aft', 'aav_status', 'ttf', 'sp', 'recline_angle'
    ]
    
    # 损伤标量，注意：distribution.csv 中通常是 HIC15
    required_labels = ['HIC15', 'Dmax', 'Nij']
    
    # 过滤条件
    required_filters = ['is_pulse_ok', 'is_injury_ok']
    
    all_required_cols = required_features + required_labels + required_filters
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"  错误: 文件未找到 {filepath}")
        return None
    except Exception as e:
        print(f"  错误: 加载文件时出错 {e}")
        return None
        
    # 检查所有必需列是否存在
    missing_cols = [col for col in all_required_cols if col not in df.columns]
    if missing_cols:
        # 特别处理 HIC vs HIC15 的常见命名问题
        if 'HIC15' in missing_cols and 'HIC' in df.columns:
            print(f"  警告: 未找到 'HIC15'，将使用 'HIC' 列代替。")
            df = df.rename(columns={'HIC': 'HIC15'})
            required_labels[required_labels.index('HIC15')] = 'HIC'
            missing_cols.remove('HIC15')
        
        if missing_cols:
            print(f"  错误: 文件 {filepath} 缺少必需列: {missing_cols}")
            return None

    # --- 核心过滤逻辑 ---
    original_count = len(df)
    filtered_df = df[
        (df['is_pulse_ok'] == True) & 
        (df['is_injury_ok'] == True)
    ].copy()
    filtered_count = len(filtered_df)
    
    print(f"  原始数据: {original_count} 条")
    print(f"  过滤后 (is_pulse_ok & is_injury_ok == True): {filtered_count} 条")
    
    if filtered_count == 0:
        print("  错误: 过滤后没有剩余数据。")
        return None
        
    # 添加 'source' 标签，用于绘图时区分
    filtered_df['source'] = file_label
    
    # 仅保留需要的列，防止内存占用过大
    final_cols = ['case_id'] + required_features + required_labels + ['source']
    # 再次检查 HIC/HIC15
    if 'HIC' in required_labels:
        final_cols[final_cols.index('HIC15')] = 'HIC'
        
    return filtered_df[final_cols]

def calculate_ais_levels(df):
    """计算 AIS 和 MAIS 等级"""
    if df is None:
        return None
        
    print(f"正在为 {df['source'].iloc[0]} 计算AIS等级...")
    
    # 确定 HIC 列名 (HIC 或 HIC15)
    hic_col = 'HIC' if 'HIC' in df.columns else 'HIC15'
    
    df['AIS_head'] = AIS_cal_head(df[hic_col])
    df['AIS_chest'] = AIS_cal_chest(df['Dmax'])
    df['AIS_neck'] = AIS_cal_neck(df['Nij'])
    
    df['MAIS'] = np.maximum.reduce([
        df['AIS_head'], 
        df['AIS_chest'], 
        df['AIS_neck']
    ])
    
    return df

def plot_scalar_distributions(combined_df, columns, output_dir):
    """为连续标量特征绘制 KDE 分布对比图"""
    print(f"\n正在绘制 {len(columns)} 个标量特征的分布图...")
    output_subdir = os.path.join(output_dir, "1_Scalar_Distributions")
    os.makedirs(output_subdir, exist_ok=True)
    
    for col in tqdm(columns, desc="Plotting Scalars"):
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=combined_df, x=col, hue='source', 
                    fill=True, common_norm=False, alpha=0.5)
        plt.title(f"Distribution Comparison for '{col}' (Continuous)")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, f"{col}_distribution.png"))
        plt.close()

def plot_categorical_distributions(combined_df, columns, output_dir):
    """为离散/等级特征绘制百分比直方图"""
    print(f"\n正在绘制 {len(columns)} 个离散/等级特征的分布图...")
    output_subdir = os.path.join(output_dir, "2_Categorical_Distributions")
    os.makedirs(output_subdir, exist_ok=True)
    
    for col in tqdm(columns, desc="Plotting Categoricals"):
        plt.figure(figsize=(12, 7))
        # 使用 stat='percent', multiple='dodge', common_norm=False
        # 这将显示每个 'source' 内部的百分比分布，是最好的对比方式
        sns.histplot(data=combined_df, x=col, hue='source', 
                     multiple='dodge', stat='percent', common_norm=False, 
                     discrete=True, shrink=0.8)
        
        plt.title(f"Distribution Comparison for '{col}' (Categorical/Levels)")
        plt.xlabel(col)
        plt.ylabel("Percentage (%) within each Source")
        plt.xticks(sorted(combined_df[col].unique()))
        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_subdir, f"{col}_distribution.png"))
        plt.close()

if __name__ == "__main__":
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载和过滤数据
    df_A = load_and_filter_data(FILE_A_PATH, FILE_A_LABEL)
    df_B = load_and_filter_data(FILE_B_PATH, FILE_B_LABEL)

    ######
    newdata_caseids = pd.read_excel(r"E:\课题组相关\理想项目\仿真数据库相关\distribution\Injury_labels_1023.xlsx")['case_id'].tolist()
    df_B = df_B[df_B['case_id'].isin(newdata_caseids)]
    print(f"  进一步过滤后，{FILE_B_LABEL} 有效样本数: {len(df_B)}")
    ######
    
    if df_A is None or df_B is None:
        print("\n错误：一个或两个文件加载失败，无法继续比较。")
        exit()
        
    # 2. 计算AIS等级
    df_A = calculate_ais_levels(df_A)
    df_B = calculate_ais_levels(df_B)
    
    # 3. 合并数据
    combined_df = pd.concat([df_A, df_B], ignore_index=True)
    print(f"\n数据已合并。总有效样本数: {len(combined_df)}")
    
    # 4. 定义要绘图的列
    # 连续标量 (输入特征)
    continuous_features = [
        'impact_velocity', 'impact_angle', 'overlap', 'll1', 'll2', 
        'btf', 'pp', 'plp', 'llattf', 'ptf', 'aft', 'ttf', 'sp', 'recline_angle'
    ]
    
    # 离散标量 (输入特征)
    discrete_features = [
        'occupant_type', 'lla_status', 'dz', 'aav_status'
    ]
    
    # 损伤标量 (标签)
    hic_col_name = 'HIC' if 'HIC' in combined_df.columns else 'HIC15'
    injury_scalars = [hic_col_name, 'Dmax', 'Nij']
    
    # 损伤等级 (计算得到)
    injury_levels = ['AIS_head', 'AIS_chest', 'AIS_neck', 'MAIS']
    
    # 5. 执行绘图
    plot_scalar_distributions(combined_df, continuous_features + injury_scalars, OUTPUT_DIR)
    plot_categorical_distributions(combined_df, discrete_features + injury_levels, OUTPUT_DIR)
    
    print("\n" + "="*50)
    print("对比完成！")
    print(f"所有对比图表已保存至: {os.path.abspath(OUTPUT_DIR)}")
    print("="*50)