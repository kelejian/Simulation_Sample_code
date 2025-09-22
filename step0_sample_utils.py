# -*- coding: utf-8 -*-
'''
采样相关的额外操作或者对distribution文件的操作
'''

# %% 验证和可视化采样结果
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def verify_and_visualize_params(filepath='distribution.npz', flag='VCS', param_pairs=None, output_dir='VCS_sample_verification'):
    """
    验证和可视化参数采样结果
    
    参数:
    - filepath: 数据文件路径 (.npz或.csv)
    - flag: 'VCS'(碰撞工况参数) 或 'MADYMO'(约束系统参数)
    - param_pairs: 指定的参数对列表，格式为[(param1, param2), ...]，如果为None则使用默认组合
    - output_dir: 图片保存目录
    
    返回:
    - 验证结果字典
    """
    print(f"=== 开始{flag}参数验证和可视化 ===")
    print(f"数据文件: {filepath}")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 根据文件后缀自动判断读取方式
    try:
        if filepath.endswith('.npz'):
            data_raw = np.load(filepath, allow_pickle=True)
            data = {key: data_raw[key] for key in data_raw.files}
            print("已加载NPZ格式文件")
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            data = {col: df[col].values for col in df.columns}
            print("已加载CSV格式文件")
        else:
            raise ValueError("不支持的文件格式，请使用.npz或.csv文件")
    except FileNotFoundError:
        print(f"错误：找不到文件 '{filepath}'。请确保文件名正确且文件存在。")
        return None
    
    print(f"数据包含 {len(data)} 个参数，{len(list(data.values())[0])} 个样本")
    
    # ******************************************************************************
    # 考虑排除部分case_id
    case_id_exclude = pd.read_csv(r'E:\课题组相关\理想项目\仿真数据库相关\distribution\case_ids_to_set_is_pulse_ok_False_0923.csv', header=None).squeeze().tolist()
    print(f"\n*排除的case_id数量: {len(case_id_exclude)}")
    data = pd.DataFrame(data)
    data = data[~np.isin(data['case_id'], case_id_exclude)]
    data = {col: data[col].values for col in data.columns}
    print(f"*排除指定case_id后，剩余样本数: {len(data['case_id'])}\n")
    # ******************************************************************************
    
    # 定义参数组
    if flag == 'VCS':
        # 碰撞工况参数
        params_to_check = ['impact_velocity', 'impact_angle', 'overlap']
        param_ranges = {
            'impact_velocity': (23, 65),  # 单位km/h, 允许有少量<25km/h的样本
            'impact_angle': (-45, 45),
            'overlap': (-1, 1)  # 特殊区间处理在后面
        }
        discrete_params = {}
        special_checks = ['overlap']
        
        # 默认参数对
        if param_pairs is None:
            param_pairs = [
                ('impact_velocity', 'impact_angle'),
                ('impact_velocity', 'overlap'),
                ('impact_angle', 'overlap')
            ]

    else:  # MADYMO
        # 约束系统参数
        params_to_check = ['occupant_type', 'll1', 'll2', 'btf', 'pp', 'plp', 
                          'lla_status', 'llattf', 'dz', 'ptf', 'aft', 'aav_status', 
                          'ttf', 'sp', 'recline_angle']
        param_ranges = {
            'll1': (2.0, 7.0),
            'll2': (1.5, 4.5),
            'btf': (10, 100),
            'pp': (40, 100),
            'plp': (20, 80),
            'aft': (10, 100),
            'recline_angle': (-10, 15)
        }
        discrete_params = {
            'occupant_type': [1, 2, 3],
            'lla_status': [0, 1],
            'aav_status': [0, 1],
            'dz': [1, 2, 3, 4]
        }
        special_checks = ['ll2_vs_ll1', 'aft_vs_btf', 'sp_vs_occupant', 'ptf_vs_btf', 'llattf_vs_btf', 'ttf_vs_aft']
        
        # 默认参数对
        if param_pairs is None:
            param_pairs = [
                ('ll1', 'll2'),
                ('sp', 'occupant_type'),
                ('ptf', 'btf'),
                ('llattf', 'btf'),
                ('ttf', 'aft'),
                ('aft', 'btf'),
                ('lla_status', 'llattf'),
                ('aav_status', 'ttf')
            ]

    print(f"检查参数组: {params_to_check}")
    print("-" * 60)
    
    # 验证参数
    print("--- 开始数据校验 ---")
    all_checks_passed = True
    verification_results = {}

    # 逐行检查是否有：某一行params_to_check中空值 or NAN的数量在[1, len(params_to_check)-1]，即存在部分参数缺失
    print("检查参数是否存在部分缺失情况...")
    for i in range(len(data['impact_velocity'])):
        nan_count = 0
        for param in params_to_check:
            if np.isnan(data[param][i]):
                nan_count += 1
        if 1 <= nan_count < len(params_to_check):
            print(f"警告: {flag}参数中第{i+1}行数据存在部分参数缺失: {nan_count}个NaN值")
            all_checks_passed = False
    if all_checks_passed:
        print(f"{flag}参数中所有行数据均完整或全部缺失，无部分缺失情况。")
    else:
        # 报错退出先
        print(f"{flag}参数中存在部分缺失情况，请检查数据完整性后重新采样。")
        return None

    # 1. 连续参数范围检查
    def check_continuous(param, min_val, max_val):
        if param not in data:
            print(f"  - 警告: 参数 '{param}' 不存在于数据中")
            return False
        # 跳过NaN值
        valid_data = data[param][~np.isnan(data[param])]
        if len(valid_data) == 0:
            print(f"  - 警告: 参数 '{param}' 全为NaN值")
            return True  # NaN值不算错误
        is_valid = np.all((valid_data >= min_val) & (valid_data <= max_val))
        print(f"  - 检查 '{param}': {'通过' if is_valid else '失败!!!!!!!'}")
        if not is_valid:
            invalid_count = np.sum((valid_data < min_val) | (valid_data > max_val))
            print(f"    异常值数量: {invalid_count}/{len(valid_data)}")
        return is_valid
    
    for param, (min_val, max_val) in param_ranges.items():
        if param in params_to_check:
            result = check_continuous(param, min_val, max_val)
            verification_results[param] = result
            all_checks_passed &= result
    
    # 2. 离散参数取值检查
    def check_discrete(param, allowed_values):
        if param not in data:
            print(f"  - 警告: 参数 '{param}' 不存在于数据中")
            return False
        # 跳过NaN值
        valid_data = data[param][~np.isnan(data[param])]
        if len(valid_data) == 0:
            print(f"  - 警告: 参数 '{param}' 全为NaN值")
            return True
        is_valid = np.all(np.isin(valid_data, allowed_values))
        print(f"  - 检查 '{param}': {'通过' if is_valid else '失败!!!!!!!'}")
        if not is_valid:
            invalid_values = valid_data[~np.isin(valid_data, allowed_values)]
            print(f"    异常值: {np.unique(invalid_values)}")
        return is_valid
    
    for param, allowed_values in discrete_params.items():
        if param in params_to_check:
            result = check_discrete(param, allowed_values)
            verification_results[param] = result
            all_checks_passed &= result
    
    # 3. 特殊参数检查
    if flag == 'VCS':
        # 检查重叠率特殊区间: (-1, -0.25]∪[0.25, 1]
        if 'overlap' in data:
            overlap_data = data['overlap'][~np.isnan(data['overlap'])]
            if len(overlap_data) > 0:
                # 检查是否在允许的区间内
                in_interval1 = (overlap_data > -1) & (overlap_data <= -0.25)
                in_interval2 = (overlap_data >= 0.25) & (overlap_data <= 1)
                is_overlap_valid = np.all(in_interval1 | in_interval2)
                print(f"  - 检查 'overlap' (在(-1,-0.25]∪[0.25,1]区间内): {'通过' if is_overlap_valid else '失败!!!!!!!'}")
                if not is_overlap_valid:
                    invalid_overlap = overlap_data[~(in_interval1 | in_interval2)]
                    print(f"    异常值数量: {len(invalid_overlap)}/{len(overlap_data)}")
                    print(f"    异常值范围: [{np.min(invalid_overlap):.4f}, {np.max(invalid_overlap):.4f}]")
                verification_results['overlap_special'] = is_overlap_valid
                all_checks_passed &= is_overlap_valid

        # 检查重叠率绝对值在0.25~0.3之间的样本，碰撞角度是否与重叠率异号，且碰撞角度绝对值>=30度
        if 'overlap' in data and 'impact_angle' in data:
            valid_mask = ~(np.isnan(data['overlap']) | np.isnan(data['impact_angle']))
            overlap_data = data['overlap'][valid_mask]
            impact_angle_data = data['impact_angle'][valid_mask]
            mask = (np.abs(overlap_data) >= 0.25) & (np.abs(overlap_data) < 0.3)
            if np.any(mask):
                angles_to_check = impact_angle_data[mask]
                overlaps_to_check = overlap_data[mask]

                # 检查这些被选中的样本是否都满足条件
                # 1. 角度绝对值 >= 30
                # 2. 角度符号与重叠率符号不同
                is_relation_valid = np.all(
                    (np.abs(angles_to_check) >= 30) & 
                    (np.sign(angles_to_check) != np.sign(overlaps_to_check))
                )
                
                print(f"  - 检查重叠率绝对值在0.25~0.3之间的样本的 'impact_angle' (与重叠率异号且绝对值>30度): {'通过' if is_relation_valid else '失败!!!!!!!'}")
                if not is_relation_valid:
                    # 找出具体是哪些样本不满足条件
                    failed_mask = ~((np.abs(angles_to_check) >= 30) & (np.sign(angles_to_check) != np.sign(overlaps_to_check)))
                    num_failed = np.sum(failed_mask)
                    print(f"    共有 {num_failed} 个样本不满足此项关联检查。")

                verification_results['overlap_angle_relation'] = is_relation_valid
                all_checks_passed &= is_relation_valid
            else:
                print("  - ! warning ! 无重叠率绝对值在0.25~0.3之间的样本，注意采样范围。")
                verification_results['overlap_angle_relation'] = True

    elif flag == 'MADYMO':
        # 约束系统特殊检查
        # 检查 ll2 < ll1
        if 'll2_vs_ll1' in special_checks and 'll1' in data and 'll2' in data:
            ll1_data = data['ll1'][~np.isnan(data['ll1'])]
            ll2_data = data['ll2'][~np.isnan(data['ll2'])]
            if len(ll1_data) > 0 and len(ll2_data) > 0:
                # 确保两个数组长度相同
                min_len = min(len(ll1_data), len(ll2_data))
                is_ll2_valid = np.all(ll2_data[:min_len] < ll1_data[:min_len])
                print(f"  - 检查 'll2' (小于 ll1): {'通过' if is_ll2_valid else '失败!!!!!!!'}")
                verification_results['ll2_vs_ll1'] = is_ll2_valid
                all_checks_passed &= is_ll2_valid
        
        # 检查 aft < 25 + btf
        if 'aft_vs_btf' in special_checks and 'aft' in data and 'btf' in data:
            aft_data = data['aft'][~np.isnan(data['aft'])]
            btf_data = data['btf'][~np.isnan(data['btf'])]
            if len(aft_data) > 0 and len(btf_data) > 0:
                min_len = min(len(aft_data), len(btf_data))
                is_aft_valid = np.all(aft_data[:min_len] < (25 + btf_data[:min_len]))
                print(f"  - 检查 'aft' (小于 btf + 25): {'通过' if is_aft_valid else '失败!!!!!!!'}")
                verification_results['aft_vs_btf'] = is_aft_valid
                all_checks_passed &= is_aft_valid

        # 检查座椅位置与乘员体型的依赖关系
        if 'sp_vs_occupant' in special_checks and 'sp' in data and 'occupant_type' in data:
            sp_data = data['sp']
            occupant_data = data['occupant_type']
            valid_mask = ~(np.isnan(sp_data) | np.isnan(occupant_data))
            if np.any(valid_mask):
                sp_valid = sp_data[valid_mask]
                occupant_valid = occupant_data[valid_mask]
                
                mask_5p = (occupant_valid == 1)
                mask_50p = (occupant_valid == 2) 
                mask_95p = (occupant_valid == 3)
                
                is_sp_valid = True
                if np.any(mask_5p):
                    is_sp_valid &= np.all((sp_valid[mask_5p] >= 10) & (sp_valid[mask_5p] <= 110))
                if np.any(mask_50p):
                    is_sp_valid &= np.all((sp_valid[mask_50p] >= -80) & (sp_valid[mask_50p] <= 80))
                if np.any(mask_95p):
                    is_sp_valid &= np.all((sp_valid[mask_95p] >= -110) & (sp_valid[mask_95p] <= 40))
                
                print(f"  - 检查 'sp' (与体型相关): {'通过' if is_sp_valid else '失败!!!!!!!'}")
                verification_results['sp_vs_occupant'] = is_sp_valid
                all_checks_passed &= is_sp_valid
        
        # 检查关联参数的计算关系
        if 'ptf_vs_btf' in special_checks and 'ptf' in data and 'btf' in data:
            ptf_data = data['ptf']
            btf_data = data['btf']
            valid_mask = ~(np.isnan(ptf_data) | np.isnan(btf_data))
            if np.any(valid_mask):
                is_ptf_valid = np.allclose(ptf_data[valid_mask], btf_data[valid_mask] + 7.0, rtol=1e-5)
                print(f"  - 检查 'ptf' (等于 btf + 7ms): {'通过' if is_ptf_valid else '失败!!!!!!!'}")
                verification_results['ptf_vs_btf'] = is_ptf_valid
                all_checks_passed &= is_ptf_valid
        
        if 'llattf_vs_btf' in special_checks and 'llattf' in data and 'btf' in data:
            llattf_data = data['llattf']
            btf_data = data['btf']
            valid_mask = ~(np.isnan(llattf_data) | np.isnan(btf_data))
            if np.any(valid_mask):
                llattf_valid = llattf_data[valid_mask]
                btf_valid = btf_data[valid_mask]
                is_llattf_valid = np.all((llattf_valid >= btf_valid) & (llattf_valid <= btf_valid + 100))
                print(f"  - 检查 'llattf' (在[btf, btf+100]内): {'通过' if is_llattf_valid else '失败!!!!!!!'}")
                verification_results['llattf_vs_btf'] = is_llattf_valid
                all_checks_passed &= is_llattf_valid
        
        if 'ttf_vs_aft' in special_checks and 'ttf' in data and 'aft' in data:
            ttf_data = data['ttf']
            aft_data = data['aft']
            valid_mask = ~(np.isnan(ttf_data) | np.isnan(aft_data))
            if np.any(valid_mask):
                ttf_valid = ttf_data[valid_mask]
                aft_valid = aft_data[valid_mask]
                is_ttf_valid = np.all((ttf_valid >= aft_valid) & (ttf_valid <= aft_valid + 100))
                print(f"  - 检查 'ttf' (在[aft, aft+100]内): {'通过' if is_ttf_valid else '失败!!!!!!!'}")
                verification_results['ttf_vs_aft'] = is_ttf_valid
                all_checks_passed &= is_ttf_valid
    
    print(f"\n--- 校验总结: {'所有检查均已通过！' if all_checks_passed else '存在未通过的检查项！'} ---\n")
    
    if not all_checks_passed:
        print("由于校验失败，将跳过可视化部分。")
        return verification_results
    
    print("--- 开始生成可视化图表 ---")
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 3.1 各参数的一维分布图
    print("正在生成一维分布图...")
    available_params = [p for p in params_to_check if p in data and not np.all(np.isnan(data[p]))]
    
    if available_params:
        n_params = len(available_params)
        n_cols = min(4, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig1.suptitle(f'{flag}参数组一维分布图', fontsize=16)
        
        if n_params == 1:
            axes1 = [axes1]
        elif n_rows == 1:
            axes1 = list(axes1) if n_cols > 1 else [axes1]
        else:
            axes1 = axes1.flatten()

        # 纵轴设为频率
        for i, param in enumerate(available_params):
            param_data = data[param][~np.isnan(data[param])]
            if len(param_data) > 0:

                if param in discrete_params:
                    sns.histplot(param_data, ax=axes1[i], stat="density", color='blue', edgecolor='black')

                else:
                    bins=20
                    if param == 'impact_velocity':
                        bins = np.arange(25, 70, 5) 
                    if param == 'impact_angle':
                        bins = np.arange(-45, 50, 5)
                    if param == 'overlap':
                        bins = np.arange(-1.0, 1.1, 0.1)
                    sns.histplot(param_data, kde=True, ax=axes1[i], stat="density", bins=bins, color='blue', edgecolor='black')

                axes1[i].set_title(f'{param}分布')
                axes1[i].set_xlabel('值')
                axes1[i].set_ylabel('频率')

        # 隐藏多余的子图
        for i in range(n_params, len(axes1)):
            axes1[i].set_visible(False)
        
        plt.tight_layout()
        dist_filename = os.path.join(output_dir, f'{flag}_parameter_distributions.png')
        plt.savefig(dist_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"一维分布图已保存: {dist_filename}")
    
    # 3.2 指定参数对的二维散点图
    print("正在生成二维散点图...")
    available_pairs = [(p1, p2) for p1, p2 in param_pairs 
                      if p1 in data and p2 in data 
                      and not np.all(np.isnan(data[p1])) 
                      and not np.all(np.isnan(data[p2]))]
    
    if available_pairs:
        n_pairs = len(available_pairs)
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        fig2.suptitle(f'{flag}参数组二维散点图', fontsize=16)
        
        if n_pairs == 1:
            axes2 = [axes2]
        elif n_rows == 1:
            axes2 = list(axes2) if n_cols > 1 else [axes2]
        else:
            axes2 = axes2.flatten()
        
        for i, (param1, param2) in enumerate(available_pairs):
            data1 = data[param1]
            data2 = data[param2]
            # 找到两个参数都不是NaN的索引
            valid_mask = ~(np.isnan(data1) | np.isnan(data2))
            if np.any(valid_mask):
                axes2[i].scatter(data1[valid_mask], data2[valid_mask], alpha=0.6, s=20)
                axes2[i].set_title(f'{param1} vs {param2}')
                axes2[i].set_xlabel(param1)
                axes2[i].set_ylabel(param2)
                axes2[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_pairs, len(axes2)):
            axes2[i].set_visible(False)
        
        plt.tight_layout()
        scatter_filename = os.path.join(output_dir, f'{flag}_parameter_scatter_plots.png')
        plt.savefig(scatter_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"二维散点图已保存: {scatter_filename}")
    
    # 3.3 碰撞工况参数三维散点图（仅VCS模式）
    if flag == 'VCS':
        print("正在生成三维散点图...")
        vcs_params = ['impact_velocity', 'impact_angle', 'overlap']
        if all(p in data and not np.all(np.isnan(data[p])) for p in vcs_params):
            # 找到三个参数都不是NaN的索引
            valid_mask = ~(np.isnan(data['impact_velocity']) | 
                          np.isnan(data['impact_angle']) | 
                          np.isnan(data['overlap']))
            
            if np.any(valid_mask):
                fig3 = plt.figure(figsize=(12, 10))
                ax3 = fig3.add_subplot(111, projection='3d')
                
                velocity_valid = data['impact_velocity'][valid_mask]
                angle_valid = data['impact_angle'][valid_mask]
                overlap_valid = data['overlap'][valid_mask]
                
                ax3.scatter(velocity_valid, angle_valid, overlap_valid, alpha=0.6, s=30)
                ax3.set_title('碰撞工况参数三维散点图', fontsize=16)
                ax3.set_xlabel('Impact Velocity (km/h)')
                ax3.set_ylabel('Impact Angle (°)')
                ax3.set_zlabel('Overlap')
                
                plt.tight_layout()
                scatter3d_filename = os.path.join(output_dir, f'{flag}_3D_scatter_plot.png')
                plt.savefig(scatter3d_filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"三维散点图已保存: {scatter3d_filename}")
    
    print(f"\n=== {flag}参数验证和可视化完成 ===")
    print(f"图片保存目录: {output_dir}")
    
    return verification_results

if __name__ == '__main__':

    verify_and_visualize_params(r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0923_V2.csv', flag='VCS', output_dir='VCS_sample_verification_0923_V2', param_pairs=[('impact_velocity', 'impact_angle'), ('impact_velocity', 'overlap'), ('impact_angle', 'overlap')])
    # verify_and_visualize_params(r'./distribution_full_test.csv', flag='MADYMO', output_dir='MADYMO_sample_verification',
    # param_pairs=[
    #         ('ll1', 'll2'),
    #         ('sp', 'occupant_type'),
    #         ('ptf', 'btf'),
    #         ('llattf', 'btf'),
    #         ('ttf', 'aft'),
    #         ('aft', 'btf'),
    #         ('lla_status', 'llattf'),
    #         ('aav_status', 'ttf')
    #         ]
    #         )

# %% 分析碰撞工况参数三维空间填充质量
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.stats import qmc, kstest, chi2
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_space_filling_quality_comprehensive(data):
    """
    全面分析碰撞工况参数三维空间填充质量 - 严格的数学验证
    """
    print("=== 严格的Sobol序列空间填充质量分析 ===")
    
    # 提取三维数据
    velocity = data['impact_velocity']
    angle = data['impact_angle'] 
    btf = data['btf']
    
    # 标准化到[0,1]³单位立方体
    # 这是所有后续分析的强制要求，确保所有参数在同一尺度下比较
    velocity_norm = (velocity - 25) / (65 - 25)
    angle_norm = (angle - (-60)) / (60 - (-60))
    btf_norm = (btf - 10) / (100 - 10)
    
    points = np.column_stack([velocity_norm, angle_norm, btf_norm])
    n_points = len(points)
    
    print(f"分析维度: 3D, 样本数: {n_points}")
    print("-" * 60)

    # --- 1. 星盘差异度 (Star Discrepancy) ---
    print("\n[指标1: 星盘差异度 (Star Discrepancy)]")
    print("解读: 这是衡量点集均匀性的黄金标准。值越接近0，代表点集在空间中的分布越均匀。\n"
          "      我们将计算Sobol样本的差异度，并与一个纯随机样本对比，以凸显其优势。")
    
    sobol_discrepancy = qmc.discrepancy(points, method='CD')
    print(f"  - Sobol样本的差异度: {sobol_discrepancy:.6f}")
    
    # 创建一个同样大小的随机样本作为对比基准
    random_points = np.random.rand(n_points, 3)
    random_discrepancy = qmc.discrepancy(random_points)
    print(f"  - 对比用随机样本的差异度: {random_discrepancy:.6f}")
    
    if sobol_discrepancy < random_discrepancy / 2:
        print("  \n结论: ✅ Sobol样本的差异度显著低于随机样本，证明其空间填充质量非常高。")
    else:
        print("  \n结论: ⚠️ Sobol样本的差异度与随机样本相比优势不明显，请检查采样过程。")
    print("-" * 60)

    # --- 2. 单维度投影的 Kolmogorov-Smirnov (K-S) 检验 ---
    print("\n[指标2: 单维度 K-S 检验]")
    print("解读: 此检验用于判断单个参数的样本分布是否符合理想的均匀分布。\n"
          "      我们会看p-value。如果p-value > 0.05，我们就有信心认为该参数的采样是均匀的。")
    
    param_names = ['速度 (Velocity)', '角度 (Angle)', 'BTF']
    all_ks_passed = True
    for i, name in enumerate(param_names):
        stat, pvalue = kstest(points[:, i], 'uniform')
        print(f"  - {name} 投影的 K-S 检验: p-value = {pvalue:.4f}")
        if pvalue <= 0.05:
            all_ks_passed = False
            print(f"    警告: {name}的p-value过低，其一维分布的均匀性不佳！")

    if all_ks_passed:
        print("  \n结论: ✅ 所有参数的一维投影均通过了均匀性检验。")
    else:
        print("  \n结论: ❌ 部分参数未通过均匀性检验，采样可能存在问题。")
    print("-" * 60)

    # --- 3. 多维卡方 (Chi-Squared) 检验 ---
    print("\n[指标3: 多维卡方 (Chi-Squared) 检验]")
    print("解读: 此检验将三维空间划分为多个小方格，检查样本点是否均匀地落入每个格子中。\n"
          "      同样，如果p-value > 0.05，说明从整体密度来看，样本是均匀分布的。")
    
    # 选择合适的网格划分数k，使得每个小方格的期望点数不低于5
    k = 0
    for k_test in range(10, 2, -1):
        if n_points / (k_test**3) >= 5.0:
            k = k_test
            break
    
    if k == 0:
        print("  - 样本量过小，无法进行有效的卡方检验。跳过此项。")
    else:
        M = k**3
        expected_freq = n_points / M
        print(f"  - 空间被划分为 {k}x{k}x{k} = {M} 个小方格，每个格子期望点数: {expected_freq:.2f}")

        observed_freq, _ = np.histogramdd(points, bins=k, range=[(0, 1), (0, 1), (0, 1)])
        
        chi2_stat = np.sum((observed_freq.flatten() - expected_freq)**2 / expected_freq)
        df = M - 1
        p_value = chi2.sf(chi2_stat, df) # sf是生存函数，等价于 1 - cdf

        print(f"  - 卡方检验统计量: {chi2_stat:.2f}, p-value = {p_value:.4f}")

        if p_value > 0.05:
            print("  \n结论: ✅ 卡方检验通过，样本点的空间密度分布是均匀的。")
        else:
            print("  \n结论: ❌ 卡方检验未通过，样本点在空间中可能存在聚集或稀疏区域。")
    print("-" * 60)
    
    # --- 4. 最近邻距离分析 (Nearest-Neighbor Distance Analysis) ---
    print("\n[指标4: 最近邻距离分析 (可视化)]")
    print("解读: 均匀分布的点集，其点与点之间的距离会比较规整。\n"
          "      如果图中出现一个非常靠近0的尖峰，说明存在点聚集的情况。Sobol序列的分布通常比\n"
          "      随机序列更窄、更集中，表明其结构更规整，没有意外的“洞”或“团”。")

    # 计算Sobol样本的最近邻距离
    nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(points)
    distances, _ = nn.kneighbors(points)
    sobol_nn_distances = distances[:, 1]

    # 计算随机样本的最近邻距离用于对比
    nn_random = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(random_points)
    distances_random, _ = nn_random.kneighbors(random_points)
    random_nn_distances = distances_random[:, 1]
    
    print(f"  - Sobol样本最近邻距离: 平均值={np.mean(sobol_nn_distances):.4f}, 标准差={np.std(sobol_nn_distances):.4f}")
    print(f"  - 随机样本最近邻距离: 平均值={np.mean(random_nn_distances):.4f}, 标准差={np.std(random_nn_distances):.4f}")
    
    # 绘图
    plt.figure(figsize=(12, 7))
    sns.kdeplot(sobol_nn_distances, label=f'Sobol Sample (std={np.std(sobol_nn_distances):.3f})', fill=True)
    sns.kdeplot(random_nn_distances, label=f'Random Sample (std={np.std(random_nn_distances):.3f})', fill=True, alpha=0.7)
    plt.title('最近邻距离分布对比 (Sobol vs. 随机)', fontsize=16)
    plt.xlabel('到最近邻居的距离', fontsize=12)
    plt.ylabel('密度', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    print("\n  结论: ✅ 请观察上图。Sobol序列的分布曲线更窄，表明其点间距更一致、结构更规整。")
    print("-" * 60)

data = np.load('distribution_NEW360.npz')
# 评估 Sobol 序列的空间填充质量
analyze_space_filling_quality_comprehensive(data)

 
# %% 简单随机采样
import numpy as np
from scipy.stats import qmc

# --- 简单随机采样版本 ---
def generate_random_samples(num_samples=1800, seed=2025):
    """
    使用简单随机采样生成参数样本，用于与Sobol采样对比
    """
    # 设置随机种子确保结果可重复
    np.random.seed(seed)
    
    print(f"开始生成 {num_samples} 个随机样本...")
    
    # 创建一个字典来存储最终的参数值
    results = {}
    
    # 直接生成所需数量的随机样本
    for i in range(num_samples):
        # 生成17个[0,1)范围内的随机数
        sample = np.random.uniform(0, 1, 17)
        
        # --- 碰撞工况参数 ---
        results.setdefault('impact_velocity', []).append(sample[0] * (65 - 25) + 25)
        results.setdefault('impact_angle', []).append(sample[1] * (60 - (-60)) + (-60))
        
        # 特殊处理重叠率
        overlap_val = sample[2] * 200 - 100  # 映射到 [-100, 100]
        # 根据备注: "如果恰好取到0附近的值或-100%，直接设为100%"
        if abs(overlap_val) < 1e-6 or np.isclose(overlap_val, -100.0):
            overlap_val = 100.0
        results.setdefault('overlap', []).append(overlap_val)

        # --- 乘员体征参数 ---
        # 映射到 [1, 2, 3]
        occupant_type = np.floor(sample[3] * 3) + 1
        results.setdefault('occupant_type', []).append(int(occupant_type))

        # --- 安全带系统 ---
        # 使用拒绝采样确保 ll2 < ll1
        attempts = 0
        max_attempts = 1000  # 防止无限循环
        while attempts < max_attempts:
            # 生成新的随机数用于限力值
            ll1_rand = np.random.uniform(0, 1)
            ll2_rand = np.random.uniform(0, 1)
            
            ll1_candidate = ll1_rand * (7.0 - 2.0) + 2.0
            ll2_candidate = ll2_rand * (4.5 - 1.5) + 1.5

            # 检查候选点是否满足约束
            if ll1_candidate > ll2_candidate:
                ll1_val = ll1_candidate
                ll2_val = ll2_candidate
                break
            attempts += 1
        
        if attempts >= max_attempts:
            # 如果拒绝采样失败，使用条件采样作为后备
            ll1_val = sample[4] * (7.0 - 2.0) + 2.0
            ll2_upper_bound = min(4.5, ll1_val)
            ll2_val = sample[5] * (ll2_upper_bound - 1.5) + 1.5

        results.setdefault('ll1', []).append(ll1_val)
        results.setdefault('ll2', []).append(ll2_val)

        btf_val = sample[6] * (100 - 10) + 10
        results.setdefault('btf', []).append(btf_val)
        results.setdefault('pp', []).append(sample[7] * (100 - 40) + 40)
        results.setdefault('plp', []).append(sample[8] * (80 - 20) + 20)
        # 映射到 [0, 1]
        results.setdefault('lla_status', []).append(int(np.floor(sample[9] * 2)))
        # 计算LLATTF
        llattf_offset_val = sample[10] * 100
        results.setdefault('llattf', []).append(btf_val + llattf_offset_val)
        # 映射到 [1, 2, 3, 4]
        results.setdefault('dz', []).append(int(np.floor(sample[11] * 4) + 1))
        # 计算PTF (确定性)
        results.setdefault('ptf', []).append(btf_val + 7.0)

        # --- 气囊系统 ---
        aft_val = sample[12] * (100 - 10) + 10
        results.setdefault('aft', []).append(aft_val)
        # 映射到 [0, 1]
        results.setdefault('aav_status', []).append(int(np.floor(sample[13] * 2)))
        # 计算TTF
        ttf_offset_val = sample[14] * 100
        results.setdefault('ttf', []).append(aft_val + ttf_offset_val)

        # --- 座椅参数 ---
        # 根据乘员体型决定座椅位置范围
        sp_sample = sample[15]
        if occupant_type == 1:  # 5% 假人
            sp_val = sp_sample * (110 - 10) + 10
        elif occupant_type == 2:  # 50% 假人
            sp_val = sp_sample * (80 - (-80)) + (-80)
        else:  # 95% 假人
            sp_val = sp_sample * (40 - (-110)) + (-110)
        results.setdefault('sp', []).append(sp_val)
        results.setdefault('recline_angle', []).append(sample[16] * (15 - (-10)) + (-10))

    # 将列表转换为Numpy数组
    for key in results:
        results[key] = np.array(results[key])

    return results

# 生成随机采样数据
print("=== 生成简单随机采样数据 ===")
random_results = generate_random_samples(num_samples=1800, seed=2025)

# 保存为 .npz 文件
random_output_filename = 'distribution_Random.npz'
np.savez_compressed(random_output_filename, **random_results)

print(f"随机采样完成! {len(random_results['impact_velocity'])}个样本点已保存至 '{random_output_filename}', 包含{len(random_results)}个参数:")
for key in random_results:
    print(f"  - {key}")

# 打印一个样本作为示例
print("\n--- 随机采样结果示例 (第一个样本点) ---")
for key, value in random_results.items():
    print(f"{key:<20}: {value[0]:.4f}")



# %% 额外在最开始增加若干组对称case
# 比如: 速度为v,角度为a>0,重叠率为o<0。则其对称case: 速度为v,角度为a-90°,重叠率为o+1；当速度为v,角度为a<0,重叠率为o>0时，其对称case: 速度为v,角度为a+90°,重叠率为o-1。如此构造若干组对称case，速度和重叠率范围同前，但角度绝对值范围为30°~60°。
import numpy as np
import pandas as pd
distributions = np.load('distribution_VCSonly.npz', allow_pickle=True)
# 生成对称case
symmetrical_cases = []
for i in range(100):
    v = np.random.uniform(30, 45)
    a = np.random.uniform(30, 60) * np.sign(np.random.randn())
    o = np.random.uniform(0.02, 0.98) * np.sign(-a) # 角度为正时重叠率为负，角度为负时重叠率为正
    is_bad = None
    case_id = 6000 + i*2 + 1
    symmetrical_cases.append((v, a, o, case_id, is_bad))  # 注意顺序：case_id在is_bad之前
    if a > 0:
        symmetrical_cases.append((v, a - 90, o + 1, case_id+1, is_bad))
    else:
        symmetrical_cases.append((v, a + 90, o - 1, case_id+1, is_bad))

# 将生成的对称case添加到主数据集的末尾
symmetrical_df = pd.DataFrame(symmetrical_cases, columns=['impact_velocity', 'impact_angle', 'overlap', 'case_id', 'is_bad'])
print("对称cases:")
print(symmetrical_df)

# 正确创建原始数据的DataFrame，确保列名顺序一致
original_df = pd.DataFrame({
    'impact_velocity': distributions['impact_velocity'],
    'impact_angle': distributions['impact_angle'], 
    'overlap': distributions['overlap'],
    'case_id': distributions['case_id'],
    'is_bad': distributions['is_bad']
})

# 合并数据
df = pd.concat([original_df, symmetrical_df], ignore_index=True)

# 打印caseid大于6000的前10行
print("\ncase_id > 6000的前10行:")
print(df[df['case_id'] > 6000].head(10))

# 保存
output_filename = 'distribution_VCSonly_with_symmetrical.npz'
print(f"正在将所有样本打包保存到 {output_filename} ...")

# 使用 np.savez_compressed 来创建一个压缩的 .npz 文件
# 文件中的每个数组都以关键字参数命名
np.savez_compressed(
    output_filename,
    impact_velocity=df['impact_velocity'].to_numpy(),
    impact_angle=df['impact_angle'].to_numpy(),
    overlap=df['overlap'].to_numpy(),
    case_id=df['case_id'].to_numpy(),
    is_bad=df['is_bad'].to_numpy(dtype=object) # 指定dtype=object以正确保存None值
)

print(f"文件 '{output_filename}' 保存成功。")
# %% 对比分析函数
import numpy as np
import matplotlib.pyplot as plt

def compare_sampling_methods():
    """
    对比Sobol采样和随机采样的效果
    """
    print("\n" + "="*80)
    print("=== Sobol采样 vs 随机采样对比分析 ===")
    print("="*80)
    
    # 加载两个数据集
    try:
        sobol_data = np.load('distribution.npz')
        random_data = np.load('distribution_Random.npz')
        print("✓ 成功加载两个数据集")
    except FileNotFoundError as e:
        print(f"✗ 文件加载失败: {e}")
        return
    
    # 提取关键的三维参数进行对比
    print("\n--- 碰撞工况参数三维空间填充质量对比 ---")
    
    def extract_normalized_3d_points(data, label):
        """提取并标准化三维点"""
        velocity = data['impact_velocity']
        angle = data['impact_angle']
        btf = data['btf']
        
        velocity_norm = (velocity - 25) / (65 - 25)
        angle_norm = (angle - (-60)) / (60 - (-60))
        btf_norm = (btf - 10) / (100 - 10)
        
        points = np.column_stack([velocity_norm, angle_norm, btf_norm])
        print(f"{label}: {len(points)} 个三维点")
        return points
    
    sobol_points = extract_normalized_3d_points(sobol_data, "Sobol采样")
    random_points = extract_normalized_3d_points(random_data, "随机采样")
    
    # 简单的对比指标
    print("\n--- 基本统计对比 ---")
    
    # 1. 最近邻距离对比
    from scipy.spatial.distance import pdist, squareform
    
    def analyze_nearest_neighbor(points, label):
        distances = pdist(points, metric='euclidean')
        distance_matrix = squareform(distances)
        np.fill_diagonal(distance_matrix, np.inf)
        nn_distances = np.min(distance_matrix, axis=1)
        
        mean_nn = np.mean(nn_distances)
        std_nn = np.std(nn_distances)
        cv_nn = std_nn / mean_nn
        
        print(f"{label}:")
        print(f"  最近邻距离均值: {mean_nn:.6f}")
        print(f"  最近邻距离标准差: {std_nn:.6f}")
        print(f"  变异系数: {cv_nn:.6f}")
        return cv_nn
    
    sobol_cv = analyze_nearest_neighbor(sobol_points, "Sobol采样")
    random_cv = analyze_nearest_neighbor(random_points, "随机采样")
    
    print(f"\n变异系数对比 (越小越好): Sobol={sobol_cv:.6f} vs 随机={random_cv:.6f}")
    if sobol_cv < random_cv:
        print("✓ Sobol采样在最近邻距离均匀性方面表现更好")
    else:
        print("✗ 随机采样在最近邻距离均匀性方面表现更好")
    
    # 2. 边缘分布均匀性对比
    print("\n--- 边缘分布均匀性对比 (KS检验) ---")
    from scipy.stats import ks_2samp
    
    uniform_ref = np.random.uniform(0, 1, len(sobol_points))
    dimensions = ['velocity', 'angle', 'btf']
    
    for i, dim_name in enumerate(dimensions):
        sobol_ks = ks_2samp(sobol_points[:, i], uniform_ref)[1]
        random_ks = ks_2samp(random_points[:, i], uniform_ref)[1]
        
        print(f"{dim_name}:")
        print(f"  Sobol p值: {sobol_ks:.6f}")
        print(f"  随机 p值: {random_ks:.6f}")
        
        if sobol_ks > random_ks:
            print(f"  ✓ Sobol采样更接近均匀分布")
        else:
            print(f"  ✗ 随机采样更接近均匀分布")
    
    # 3. 网格覆盖对比
    print("\n--- 网格覆盖对比 ---")
    
    def analyze_grid_coverage(points, label, grid_size=10):
        grid_counts = np.zeros((grid_size, grid_size, grid_size))
        grid_indices = np.floor(points * grid_size).astype(int)
        grid_indices = np.clip(grid_indices, 0, grid_size - 1)
        
        for i in range(len(points)):
            x, y, z = grid_indices[i]
            grid_counts[x, y, z] += 1
        
        occupied_grids = np.sum(grid_counts > 0)
        total_grids = grid_size ** 3
        occupancy_rate = occupied_grids / total_grids
        
        print(f"{label}:")
        print(f"  网格占用率: {occupancy_rate:.1%}")
        print(f"  占用网格数: {occupied_grids}/{total_grids}")
        
        return occupancy_rate
    
    sobol_occupancy = analyze_grid_coverage(sobol_points, "Sobol采样")
    random_occupancy = analyze_grid_coverage(random_points, "随机采样")
    
    if sobol_occupancy > random_occupancy:
        print("✓ Sobol采样具有更好的网格覆盖率")
    else:
        print("✗ 随机采样具有更好的网格覆盖率")
    
    # 4. 可视化对比
    print("\n--- 生成对比可视化 ---")
    
    fig = plt.figure(figsize=(20, 10))
    
    # Sobol采样3D散点图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(sobol_points[:, 0], sobol_points[:, 1], sobol_points[:, 2], 
               alpha=0.6, s=20, c='blue')
    ax1.set_title('Sobol采样 - 三维空间分布', fontsize=16)
    ax1.set_xlabel('Velocity (normalized)')
    ax1.set_ylabel('Angle (normalized)')
    ax1.set_zlabel('BTF (normalized)')
    
    # 随机采样3D散点图
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2], 
               alpha=0.6, s=20, c='red')
    ax2.set_title('随机采样 - 三维空间分布', fontsize=16)
    ax2.set_xlabel('Velocity (normalized)')
    ax2.set_ylabel('Angle (normalized)')
    ax2.set_zlabel('BTF (normalized)')
    
    plt.tight_layout()
    plt.show()
    
    # 生成2D投影对比
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig2.suptitle('二维投影对比 - 上行：Sobol采样，下行：随机采样', fontsize=16)
    
    # 三个二维投影
    projections = [
        (0, 1, 'Velocity vs Angle'),
        (0, 2, 'Velocity vs BTF'),
        (1, 2, 'Angle vs BTF')
    ]
    
    for i, (dim1, dim2, title) in enumerate(projections):
        # Sobol采样
        axes[0, i].scatter(sobol_points[:, dim1], sobol_points[:, dim2], 
                          alpha=0.6, s=15, c='blue')
        axes[0, i].set_title(f'Sobol - {title}')
        axes[0, i].grid(True, alpha=0.3)
        
        # 随机采样
        axes[1, i].scatter(random_points[:, dim1], random_points[:, dim2], 
                          alpha=0.6, s=15, c='red')
        axes[1, i].set_title(f'Random - {title}')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== 对比总结 ===")
    print("理论上，Sobol采样应该在以下方面表现更好：")
    print("1. 更均匀的空间分布（更低的最近邻距离变异系数）")
    print("2. 更好的维度覆盖（更高的网格占用率）")
    print("3. 更稳定的边缘分布")
    print("4. 更好的低差异性质")
    
    return {
        'sobol_cv': sobol_cv,
        'random_cv': random_cv,
        'sobol_occupancy': sobol_occupancy,
        'random_occupancy': random_occupancy
    }

# 运行对比分析
if __name__ == '__main__':
    comparison_results = compare_sampling_methods()

# %% 给distribution（包含众多array）添加新的两列：'case_id' 和 'is_bad'; 之后重新保存
import numpy as np
import pandas as pd

# 加载 .npz 文件
distribution = np.load('distribution_0823_final.npz', allow_pickle=True)
print("NPZ文件包含的键名:", distribution.files)

# 检查数据结构
print("\n各数组的形状:")
for key in distribution.files:
    print(f"{key}: {distribution[key].shape}")

# 正确的转换方法：手动构建字典
data_dict = {}
for key in distribution.files:
    data_dict[key] = distribution[key]

df = pd.DataFrame(data_dict)
print("\n转换成功!")
print(f"DataFrame 形状: {df.shape}")
print(df.head(5))

# 添加新列
df['case_id'] = df.index + 1
df['is_bad'] = False

# 显示添加新列后的结果
print(f"\n添加新列后的 DataFrame 形状: {df.shape}")
print("新列预览:")
print(df[['case_id', 'is_bad']].head())

# 保存为 CSV
df.to_csv('distribution_0825_final.csv', index=False)

# 保存为 NPZ - 确保键名为字符串
np.savez_compressed('distribution_0825_final.npz', 
                   **{str(col): df[col].values for col in df.columns})

print("\n保存完成!")
# %% distribution.npz转换为表格，包含中文参数名
import numpy as np
import pandas as pd

def convert_npz_to_table(npz_file='distribution.npz', output_excel='simulation_parameters.xlsx', output_csv='simulation_parameters.csv'):
    """
    将distribution.npz文件转换为表格，包含中文参数名
    """
    # 读取npz文件
    data = np.load(npz_file, allow_pickle=True)
    
    # 参数中文名映射
    param_chinese_names = {
        'impact_velocity': '碰撞速度(km/h)',
        'impact_angle': '碰撞角度(°)', 
        'overlap': '重叠率',
        'occupant_type': '乘员体型(1:5%, 2:50%, 3:95%)',
        'll1': '一级限力值(kN)',
        'll2': '二级限力值(kN)', 
        'btf': '预紧器点火时刻(ms)',
        'pp': '预紧器抽入量(mm)',
        'plp': '腰部预紧抽入量(mm)',
        'lla_status': '二级限力切换状态(0/1)',
        'llattf': '二级限力切换时刻(ms)',
        'dz': 'D环高度(1-4)',
        'ptf': '预紧器释放时刻(ms)', 
        'aft': '气囊点火时刻(ms)',
        'aav_status': '二级主动泄气孔状态(0/1)',
        'ttf': '二级泄气孔切换时刻(ms)',
        'sp': '座椅前后位置(mm)',
        'recline_angle': '座椅靠背角度(°)',
        'case_id': '案例编号',
        'is_bad': '是否为异常案例'
    }
    
    # 创建DataFrame
    df_data = {}
    
    # 按参数类别排序并添加数据
    param_order = [
        'case_id',  # 案例编号
        'is_bad',  # 是否为异常案例
        'impact_velocity', 'impact_angle', 'overlap',  # 碰撞工况
        'occupant_type',  # 乘员参数
        'll1', 'll2', 'btf', 'pp', 'plp', 'lla_status', 'llattf', 'dz', 'ptf',  # 安全带
        'aft', 'aav_status', 'ttf',  # 气囊
        'sp', 'recline_angle'  # 座椅
    ]
    
    for param in param_order:
        if param in data:
            chinese_name = param_chinese_names.get(param, param)
            # 确保数据类型正确
            param_data = data[param]
            # 统一转换为数值类型
            param_data = pd.to_numeric(param_data, errors='coerce')
            
            # 对于整数参数，如果没有缺失值则转为整数
            if param in ['occupant_type', 'lla_status', 'dz', 'aav_status', 'case_id', 'is_bad']:
                if not pd.isna(param_data).any():
                    param_data = param_data.astype(int)
                
            # 中文+英文列名
            df_data[f"{param} ({chinese_name})"] = param_data
    
    # 创建DataFrame
    df = pd.DataFrame(df_data)
    
    # 保存文件时使用不同的方法
    excel_file = output_excel
    csv_file = output_csv
    
    # 保存为Excel文件（推荐用于查看数据）
    df.to_excel(excel_file, index=False, engine='openpyxl')
    
    # 保存CSV时不使用UTF-8-BOM，使用标准UTF-8
    df.to_csv(csv_file, index=False, encoding='utf-8', float_format='%.6g')
    
    print(f"转换完成!")
    print(f"Excel文件: {excel_file}")
    print(f"CSV文件: {csv_file}")
    print(f"总共 {len(df)} 个仿真案例，{len(df.columns)} 个参数")
    
    # 显示前5行预览
    print("\n数据预览:")
    print(df.head())
    
    return df

# 执行转换
if __name__ == '__main__':
    df = convert_npz_to_table(
        npz_file=r'I:\000 LX\dataset0715\02\distribution_VCSonly_with_symmetrical.npz',  
        output_excel='distribution_VCSonly_with_symmetrical.xlsx',
        output_csv='distribution_VCSonly_with_symmetrical.csv'
    )

# %% 把之前的旧的distribution.csv中is_pulse_ok为true的case的行替换进新的distribution.csv中
import numpy as np
import pandas as pd
import os

old_distribution_file = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0922.csv'

new_distribution_file = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0923_V0.csv'

# 读取文件
old_csv = pd.read_csv(old_distribution_file).set_index('case_id', drop=False)
new_df = pd.read_csv(new_distribution_file).set_index('case_id', drop=False)

# 旧的distribution中is_pulse_ok为true的case
old_df_filtered = old_csv[old_csv['is_pulse_ok'] == True]

print(f"旧文件总行数: {len(old_csv)}")
print(f"新文件总行数: {len(new_df)}")
print(f"旧文件中is_pulse_ok为true的行数: {len(old_df_filtered)}")

# 检查列名是否完全匹配
old_cols = set(old_df_filtered.columns)
new_cols = set(new_df.columns)
if old_cols != new_cols:
    print("警告: 两个文件的列名不完全匹配!")
    print(f"旧文件多出的列: {old_cols - new_cols}")
    print(f"新文件多出的列: {new_cols - old_cols}")
    raise ValueError("列名不匹配，无法继续替换操作。")
else:
    print("列名匹配，继续检查索引是否对其顺序。")
# 检查索引对齐
for col1, col2 in zip(old_df_filtered.columns, new_df.columns):
    if col1 != col2:
        print(f"警告: 列名顺序不匹配! 旧文件列: {col1}, 新文件列: {col2}")
        raise ValueError("列名顺序不匹配，无法继续替换操作。")
print("列名顺序匹配，继续替换操作。")

# 遍历旧的distribution中is_pulse_ok为true的case，将其行替换进新的distribution中
replace_count = 0
for case_id, row in old_df_filtered.iterrows():
    if case_id in new_df.index:
        new_df.loc[case_id] = row
        replace_count += 1
    else:
        print(f"警告: case_id {case_id} 在新的distribution中未找到，无法替换。")

print(f"总共替换了 {replace_count} 行")

# 保存新的distribution.csv
new_name = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0923.csv'
new_df.to_csv(new_name, index=False)
# %% 对比两个csv文件内容的差异
import pandas as pd
import numpy as np
file1 = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0917.csv'
file2 = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0919.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 行数相同，逐行对比内容是否一样
if df1.shape[0] != df2.shape[0]:
    print(f"行数不一致: {file1} 有 {df1.shape[0]} 行, {file2} 有 {df2.shape[0]} 行")
    differences = []
    for i in range(np.min(df1.shape[0], df2.shape[0])):
        row1 = df1.iloc[i]
        row2 = df2.iloc[i]
        if not row1.equals(row2):
            # 打印具体不同的列的值
            diff_cols = row1[row1 != row2].index.tolist()
            diff_details = {col: (row1[col], row2[col]) for col in diff_cols}
            differences.append((i+1, diff_details))  # 行号从1开始
            print(f"第 {i+1} 行不同: {diff_details}")
    
    if not differences:
        print("两个文件内容完全一致。")
    else:
        print(f"总共有 {len(differences)} 行内容不同。")
else:
    print(f"行数一致: {df1.shape[0]} 行")
    differences = []
    for i in range(df1.shape[0]):
        row1 = df1.iloc[i]
        row2 = df2.iloc[i]
        if not row1.equals(row2):
            # 打印具体不同的列的值
            diff_cols = row1[row1 != row2].index.tolist()
            diff_details = {col: (row1[col], row2[col]) for col in diff_cols}
            differences.append((i+1, diff_details))  # 行号从1开始
            print(f"第 {i+1} 行不同: {diff_details}")
    
    if not differences:
        print("两个文件内容完全一致。")
    else:
        print(f"总共有 {len(differences)} 行内容不同。")



# %% 读取指定目录下的acc的xlsx文件，仅将distribution文件中的对应行的have_run值更新为True或保持False，其它不变
import os
import pandas as pd
def update_have_run_status(acc_dir, distribution_path, new_distribution_path):
    # 读取distribution文件
    if distribution_path.endswith('.npz'):
        distribution_npz = np.load(distribution_path, allow_pickle=True)
        distribution_df = pd.DataFrame({
                key: distribution_npz[key]
                for key in distribution_npz.files
            }).set_index('case_id')
    elif distribution_path.endswith('.csv'):
        distribution_df = pd.read_csv(distribution_path)
        distribution_df.set_index('case_id', inplace=True, drop=False)
    else:
        raise ValueError("Unsupported distribution file format. Use .csv or .npz")

    # 遍历acc目录下的所有xlsx文件,形如case_{case_id}.xlsx
    change_count = 0
    for filename in os.listdir(acc_dir):
        if filename.startswith('case_') and filename.endswith('.xlsx'):
            try:
                case_id_str = filename.split('_')[1].split('.')[0]
                case_id = int(case_id_str)
                if case_id in distribution_df.index:
                    distribution_df.at[case_id, 'have_run'] = True
                    print(f"Updated have_run to True for case_id {case_id}")
                    change_count += 1
                else:
                    print(f"Warning: case_id {case_id} from file {filename} not found in distribution.")
            except (IndexError, ValueError) as e:
                print(f"Error processing file {filename}: {str(e)}")

    # 保存更新后的distribution文件
    if new_distribution_path.endswith('.npz'):
        np.savez(new_distribution_path, **{col: distribution_df[col].values for col in distribution_df.columns})
    elif new_distribution_path.endswith('.csv'):
        distribution_df.to_csv(new_distribution_path, index=False)
    else:
        raise ValueError("Unsupported new distribution file format. Use .csv or .npz")

    print(f"Total cases updated with have_run=True: {change_count}")
    print(f"Updated distribution file saved to {new_distribution_path}")

xlsx_results_dir = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\new模型_全宽正碰结果\acc_results'
distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution_0917.csv'
new_distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution_0917_updated_have_run.csv'

update_have_run_status(xlsx_results_dir, distribution_path, new_distribution_path)

# %% 读取指定目录下的acc的csv文件，将distribution文件中的对应行的is_pulse_ok值更新为True或False。注意该目录下必须已经是干净的acc文件
import os
import pandas as pd
def update_is_pulse_ok_status(acc_dir, distribution_path, new_distribution_path):
    # 读取distribution文件
    if distribution_path.endswith('.npz'):
        distribution_npz = np.load(distribution_path, allow_pickle=True)
        distribution_df = pd.DataFrame({
                key: distribution_npz[key]
                for key in distribution_npz.files
            }).set_index('case_id', drop=False)
    elif distribution_path.endswith('.csv'):
        distribution_df = pd.read_csv(distribution_path)
        distribution_df.set_index('case_id', inplace=True, drop=False)
    else:
        raise ValueError("Unsupported distribution file format. Use .csv or .npz")

    # 遍历acc目录下的所有x开头的csv文件，形如x{case_id}.csv
    change_count = 0
    for filename in os.listdir(acc_dir):
        if filename.startswith('x') and filename.endswith('.csv'):
            try:
                case_id_str = filename.split('x')[1].split('.')[0]
                case_id = int(case_id_str)
                if case_id in distribution_df.index:
                    distribution_df.at[case_id, 'have_run'] = True
                    distribution_df.at[case_id, 'is_pulse_ok'] = True
                    print(f"Updated is_pulse_ok to True for case_id {case_id}")
                    change_count += 1
                else:
                    print(f"Warning: case_id {case_id} from file {filename} not found in distribution.")
            except (IndexError, ValueError) as e:
                print(f"Error processing file {filename}: {str(e)}")

    # 保存更新后的distribution文件
    if new_distribution_path.endswith('.npz'):
        np.savez(new_distribution_path, **{col: distribution_df[col].values for col in distribution_df.columns})
    elif new_distribution_path.endswith('.csv'):
        distribution_df.to_csv(new_distribution_path, index=False)
    else:
        raise ValueError("Unsupported new distribution file format. Use .csv or .npz")
    
    print(f"Total cases updated with is_pulse_ok=True: {change_count}")
    print(f"Updated distribution file saved to {new_distribution_path}")

acc_data_dir = r'I:\000 LX\dataset0715\03\acc_data_0918_470'
distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution_0917_updated_have_run.csv'
new_distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution_0917_final.csv'

update_is_pulse_ok_status(acc_data_dir, distribution_path, new_distribution_path)
# %% 额外将部分case的is_pulse_ok改为False。这部分case的csv文件暂时保留
import os
import pandas as pd
import numpy as np

distribution_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0923_V1.csv'

# 读取distribution文件
if distribution_path.endswith('.npz'):
    distribution_npz = np.load(distribution_path, allow_pickle=True)
    distribution_df = pd.DataFrame({
            key: distribution_npz[key]
            for key in distribution_npz.files
        }).set_index('case_id', drop=False)
elif distribution_path.endswith('.csv'):
    distribution_df = pd.read_csv(distribution_path)
    distribution_df.set_index('case_id', inplace=True, drop=False)
else:
    raise ValueError("Unsupported distribution file format. Use .csv or .npz")

# 重叠率绝对值小于0.25的case的is_pulse_ok改为False
# 重叠率绝对值在0.25~0.3之间的case中：碰撞角度是否与重叠率异号，且碰撞角度绝对值>=30度满足条件，其is_pulse_ok保持True，否则改为False
mask1 = (abs(distribution_df['overlap']) < 0.25)
mask2 = (abs(distribution_df['overlap']) >= 0.25) & (abs(distribution_df['overlap']) < 0.3) & ((abs(distribution_df['impact_angle']) < 30) | (np.sign(distribution_df['impact_angle']) == np.sign(distribution_df['overlap'])))
old_df_filtered = distribution_df[mask1 | mask2]
print(f"不符合条件的行数: {len(old_df_filtered)}")
print(old_df_filtered[['case_id', 'impact_angle', 'overlap', 'is_pulse_ok']])

case_ids_to_update = old_df_filtered['case_id'].tolist()
for case_id in case_ids_to_update:
    distribution_df.at[case_id, 'is_pulse_ok'] = False
    print(f"Updated is_pulse_ok to False for case_id {case_id}.")
# 将case_ids_to_update保存为csv文件
case_ids_df = pd.DataFrame(case_ids_to_update)
case_ids_df.to_csv(r'E:\课题组相关\理想项目\仿真数据库相关\distribution\case_ids_to_set_is_pulse_ok_False_0923.csv', index=False, header=None)
print("case_ids_to_update已保存为CSV文件。")

# # 保存更新后的distribution文件
# new_distribution_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0923_V2.csv'
# if new_distribution_path.endswith('.npz'):
#     np.savez(new_distribution_path, **{col: distribution_df[col].values for col in distribution_df.columns})
# elif new_distribution_path.endswith('.csv'):
#     distribution_df.to_csv(new_distribution_path, index=False)


# %%
# %%
# %%
# %%
# %%
