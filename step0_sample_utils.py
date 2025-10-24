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
    # case_id_exclude = pd.read_csv(r'E:\课题组相关\理想项目\仿真数据库相关\distribution\case_ids_to_set_is_pulse_ok_False_0923.csv', header=None).squeeze().tolist()
    # print(f"\n*排除的case_id数量: {len(case_id_exclude)}")
    # data = pd.DataFrame(data)
    # data = data[~np.isin(data['case_id'], case_id_exclude)]
    # data = {col: data[col].values for col in data.columns}
    # print(f"*排除指定case_id后，剩余样本数: {len(data['case_id'])}\n")
    # 排除is_pulse_ok为False的case_id, 不包含is_pulse_ok为NaN的样本
    if 'case_id' in data and 'is_pulse_ok' in data:
        print("\n*排除is_pulse_ok为False的case_id, 但不排除is_pulse_ok为NaN的样本")
        data_df = pd.DataFrame(data)
        initial_count = len(data_df)
        data_df = data_df[~(data_df['is_pulse_ok'] == False)]
        filtered_count = len(data_df)
        data = {col: data_df[col].values for col in data_df.columns}
        print(f"*排除is_pulse_ok为False后，剩余样本数: {filtered_count} (初始样本数: {initial_count})\n")

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
                print(f"  - 检查 'overlap' 在(-1,-0.25]∪[0.25,1]区间内: {'通过' if is_overlap_valid else '失败!!!!!!!'}")
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
        
        # 检查 aft < 25 + btf, 只考虑case_id>1000的样本
        if 'aft_vs_btf' in special_checks and 'aft' in data and 'btf' in data:
            ##########################################################################################################################
            if 'case_id' in data:
                aft_data = data['aft'][(data['case_id'] > 1000) & (~np.isnan(data['aft']))]
                btf_data = data['btf'][(data['case_id'] > 1000) & (~np.isnan(data['btf']))]
                print(f"  - 注意: 'aft_vs_btf' 检查仅针对 case_id > 1000 的样本，共计 {len(aft_data)} 个样本")
            else:
                aft_data = data['aft'][~np.isnan(data['aft'])]
                btf_data = data['btf'][~np.isnan(data['btf'])]
            ##########################################################################################################################
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
        
        # 检查llattf与lla_status的关系：当lla_status=0时llattf应为0，当lla_status=1时llattf应在[btf, btf+100]内
        if 'llattf_vs_btf' in special_checks and 'llattf' in data and 'btf' in data and 'lla_status' in data:
            llattf_data = data['llattf']
            btf_data = data['btf']
            lla_status_data = data['lla_status']
            valid_mask = ~(np.isnan(llattf_data) | np.isnan(btf_data) | np.isnan(lla_status_data))
            
            if np.any(valid_mask):
                llattf_valid = llattf_data[valid_mask]
                btf_valid = btf_data[valid_mask]
                lla_status_valid = lla_status_data[valid_mask]
                
                # 检查lla_status=0时llattf=0
                mask_status_0 = (lla_status_valid == 0)
                is_llattf_valid_0 = True
                if np.any(mask_status_0):
                    is_llattf_valid_0 = np.allclose(llattf_valid[mask_status_0], 0.0, atol=1e-5)
                
                # 检查lla_status=1时llattf在[btf, btf+100]内
                mask_status_1 = (lla_status_valid == 1)
                is_llattf_valid_1 = True
                if np.any(mask_status_1):
                    is_llattf_valid_1 = np.all(
                        (llattf_valid[mask_status_1] >= btf_valid[mask_status_1]) & 
                        (llattf_valid[mask_status_1] <= btf_valid[mask_status_1] + 100)
                    )
                
                is_llattf_valid = is_llattf_valid_0 and is_llattf_valid_1
                print(f"  - 检查 'llattf' (lla_status=0时为0，=1时在[btf, btf+100]内): {'通过' if is_llattf_valid else '失败!!!!!!!'}")
                verification_results['llattf_vs_btf'] = is_llattf_valid
                all_checks_passed &= is_llattf_valid
        
        # 检查ttf与aav_status的关系：当aav_status=0时ttf应为0，当aav_status=1时ttf应在[aft, aft+100]内且>0.5*btf
        if 'ttf_vs_aft' in special_checks and 'ttf' in data and 'aft' in data and 'aav_status' in data:
            ttf_data = data['ttf']
            aft_data = data['aft']
            aav_status_data = data['aav_status']
            valid_mask = ~(np.isnan(ttf_data) | np.isnan(aft_data) | np.isnan(aav_status_data))
            
            if np.any(valid_mask):
                ttf_valid = ttf_data[valid_mask]
                aft_valid = aft_data[valid_mask]
                aav_status_valid = aav_status_data[valid_mask]
                
                # 检查aav_status=0时ttf=0
                mask_status_0 = (aav_status_valid == 0)
                is_ttf_valid_0 = True
                if np.any(mask_status_0):
                    is_ttf_valid_0 = np.allclose(ttf_valid[mask_status_0], 0.0, atol=1e-5)
                
                # 检查aav_status=1时ttf在[aft, aft+100]内
                mask_status_1 = (aav_status_valid == 1)
                is_ttf_valid_1 = True
                if np.any(mask_status_1):
                    is_ttf_valid_1 = np.all(
                        (ttf_valid[mask_status_1] >= aft_valid[mask_status_1]) & 
                        (ttf_valid[mask_status_1] <= aft_valid[mask_status_1] + 100)
                    )
                    # 额外检查ttf > 0.5*btf（针对aav_status=1的情况）
                    if 'btf' in data and is_ttf_valid_1:
                        btf_data = data['btf']
                        btf_valid_mask = valid_mask & (aav_status_data == 1) & (~np.isnan(btf_data))
                        if np.any(btf_valid_mask):
                            ttf_for_check = ttf_data[btf_valid_mask]
                            btf_for_check = btf_data[btf_valid_mask]
                            is_ttf_valid_1 &= np.all(ttf_for_check > 0.5 * btf_for_check)
                
                is_ttf_valid = is_ttf_valid_0 and is_ttf_valid_1
                print(f"  - 检查 'ttf' (aav_status=0时为0，=1时在[aft, aft+100]内且>0.5*btf): {'通过' if is_ttf_valid else '失败!!!!!!!'}")
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
            #############################################################
            # 如果是aft_vs_btf检查，只考虑case_id>1000的样本
            if flag == 'MADYMO' and (param1, param2) == ('aft', 'btf') and 'case_id' in data:
                data1 = data1[data['case_id'] > 1000]
                data2 = data2[data['case_id'] > 1000]
                print(f"  - 注意: 'aft_vs_btf' 散点图仅针对 case_id > 1000 的样本，共计 {len(data1)} 个样本")
            #############################################################
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

    verify_and_visualize_params(r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_test2.csv', flag='VCS', output_dir='VCS_sample_verification_test', param_pairs=[('impact_velocity', 'impact_angle'), ('impact_velocity', 'overlap'), ('impact_angle', 'overlap')])
    verify_and_visualize_params(r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_test2.csv', flag='MADYMO', output_dir='MADYMO_sample_verification_test',
    param_pairs=[
            ('ll1', 'll2'),
            ('sp', 'occupant_type'),
            ('ptf', 'btf'),
            ('llattf', 'btf'),
            ('ttf', 'aft'),
            ('aft', 'btf'),
            ('lla_status', 'llattf'),
            ('aav_status', 'ttf')
            ]
            )



# %% 1.读取指定目录下的acc的xlsx文件，仅将distribution文件中的对应行的have_run值更新为True或保持False，其它不变
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

# %% 2.读取指定目录下的acc的csv文件，将distribution文件中的对应行的is_pulse_ok值更新为True
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

# %% 3.根据check_csv_pulse结果，将指定case_id的is_pulse_ok改为False
import pandas as pd
distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0926_V2.csv'
new_distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0926_V3.csv'
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
# 需要更新的case_id列表
case_ids_to_update = [1702, 2186]
for case_id in case_ids_to_update:
    if case_id in distribution_df.index:
        distribution_df.at[case_id, 'is_pulse_ok'] = False
        print(f"Updated is_pulse_ok to False for case_id {case_id}.")
    else:
        print(f"Warning: case_id {case_id} not found in distribution.")
# 保存更新后的distribution文件
if new_distribution_path.endswith('.npz'):
    np.savez(new_distribution_path, **{col: distribution_df[col].values for col in distribution_df.columns})
elif new_distribution_path.endswith('.csv'):
    distribution_df.to_csv(new_distribution_path, index=False)
    print("Updated distribution file has been saved.")
# %% -Final.将头颈胸损伤标签（HIC15, Dmax, Nij）添加到distribution文件中
import numpy as np
import pandas as pd
distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1015.csv'
new_distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1016.csv'
Injury_labels_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\Injury_labels_1016.xlsx'
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

# 读取injury_labels文件
injury_df = pd.read_excel(Injury_labels_path).set_index('case_id', drop=False)

injury_columns = ['HIC15', 'Dmax', 'Nij'] # Dmax原始单位为m，这里要转换为mm

# 如果distribution_df中没有injury_columns，先在DataFrame中添加这些列，初始值为NaN
for col in injury_columns:
    if col not in distribution_df.columns:
        distribution_df[col] = np.nan
        print(f"Added missing column '{col}' to distribution DataFrame.")

# 遍历injury_df，将对应case_id的injury_columns值更新到distribution_df中；并将其is_injury_ok改为True
update_count = 0
for case_id, row in injury_df.iterrows():
    if case_id in distribution_df.index:
        for col in injury_columns:
            # Dmax原始单位为m，这里要转换为mm
            if col == 'Dmax':
                distribution_df.at[case_id, col] = row[col] * 1000
            else:
                distribution_df.at[case_id, col] = row[col]
        distribution_df.at[case_id, 'is_injury_ok'] = True
        update_count += 1
    else:
        print(f"Warning: case_id {case_id} from injury_labels not found in distribution.")
print(f"Total cases updated with injury labels: {update_count}")
# injury_df的Valid列为False的case，其distribution_df中的is_injury_ok改为False
False_count = 0
for case_id, row in injury_df.iterrows():
    if case_id in distribution_df.index:
        if row['Valid'] == False:
            distribution_df.at[case_id, 'is_injury_ok'] = False
            False_count += 1
    else:
        print(f"Warning: case_id {case_id} from injury_labels not found in distribution.")
print(f"Total cases set is_injury_ok to False based on injury labels: {False_count}")
# 保存更新后的distribution文件
if new_distribution_path.endswith('.npz'):
    np.savez(new_distribution_path, **{col: distribution_df[col].values for col in distribution_df.columns})
elif new_distribution_path.endswith('.csv'):
    distribution_df.to_csv(new_distribution_path, index=False)
    print("Updated distribution file with injury labels has been saved.")
    
# %% 4.为distribution文件计算delta-v。如果没有该列，则添加该列，初始值为NaN
import numpy as np
import pandas as pd
from scipy import integrate
distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0929.csv'
new_distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0929_V2.csv'
acc_csv_dir = r'F:\VCS_acc_data\acc_data_before0928_2375'

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
# 如果distribution_df中没有delta_vx或delta_vy或delta_v列，先在DataFrame中添加该列，初始值为NaN
if 'delta_vx(kph)' not in distribution_df.columns:
    distribution_df['delta_vx(kph)'] = np.nan
    print("Added missing column 'delta_vx(kph)' to distribution DataFrame.")
if 'delta_vy(kph)' not in distribution_df.columns:
    distribution_df['delta_vy(kph)'] = np.nan
    print("Added missing column 'delta_vy(kph)' to distribution DataFrame.")
if 'delta_v(kph)' not in distribution_df.columns:
    distribution_df['delta_v(kph)'] = np.nan
    print("Added missing column 'delta_v(kph)' to distribution DataFrame.")
# 遍历acc_csv_dir目录下的所有x开头的csv文件，形如x{case_id}.csv
# 对应的y文件形如y{case_id}.csv
case_ids = []
if not os.path.isdir(acc_csv_dir):
    raise FileNotFoundError(f"Directory not found: {acc_csv_dir}")
for file in os.listdir(acc_csv_dir):
    if file.startswith('x') and file.endswith('.csv'):
        try:
            case_id = int(file.split('.')[0][1:])
            case_ids.append(case_id)
        except (ValueError, IndexError):
            print(f"Warning: Could not parse case_id from filename '{file}'. Skipping.")
case_ids.sort()
print(f"Found {len(case_ids)} case_ids in acc directory.")

cal_success_count = 0
for case_id in case_ids:
    x_file = os.path.join(acc_csv_dir, f'x{case_id}.csv')
    y_file = os.path.join(acc_csv_dir, f'y{case_id}.csv')
    # 检查文件是否存在
    if not os.path.exists(y_file):
        print(f"Warning: Missing y file for case_id {case_id}. Skipping.")
        continue
    x_data = pd.read_csv(x_file, sep='\t', header=None, names=['time', 'ax']) # s和m/s²
    y_data = pd.read_csv(y_file, sep='\t', header=None, names=['time', 'ay'])
    # 计算delta_v：根据x{case_id}.csv文件积分得到delta_vx，再根据y{case_id}.csv文件积分得到delta_vy，最后计算delta_v=sqrt(delta_vx^2+delta_vy^2)
    # 使用 .to_numpy() 转换成numpy数组，效率更高
    delta_vx = integrate.simpson(y=x_data['ax'].to_numpy(), x=x_data['time'].to_numpy())
    delta_vy = integrate.simpson(y=y_data['ay'].to_numpy(), x=y_data['time'].to_numpy())
    delta_v = np.sqrt(delta_vx**2 + delta_vy**2)
    if case_id in distribution_df.index:
        distribution_df.at[case_id, 'delta_vx(kph)'] = delta_vx * 3.6  # m/s转换为kph
        distribution_df.at[case_id, 'delta_vy(kph)'] = delta_vy * 3.6  # m/s转换为kph
        distribution_df.at[case_id, 'delta_v(kph)'] = delta_v * 3.6  # m/s转换为kph
        cal_success_count += 1
        # print(f"Calculated delta_v for case_id {case_id}: delta_vx={delta_vx:.2f}, delta_vy={delta_vy:.2f}, delta_v={delta_v:.2f}")
    else:
        print(f"Warning: case_id {case_id} from acc files not found in distribution.")
print(f"Total cases with delta_v calculated: {cal_success_count}")
# 保存更新后的distribution文件
if new_distribution_path.endswith('.npz'):
    np.savez(new_distribution_path, **{col: distribution_df[col].values for col in distribution_df.columns})
elif new_distribution_path.endswith('.csv'):
    distribution_df.to_csv(new_distribution_path, index=False)
    print("Updated distribution file with delta_v has been saved.")


# %% 5.将主驾侧的have_run,is_pulse_ok,delta_vx(kph),delta_vy(kph),delta_v(kph) 的值复制到对应的副驾侧行中
import numpy as np
import pandas as pd
distribution_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1017_V2.csv'
new_distribution_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1017_final.csv'
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
# 遍历当前DataFrame，找到is_driver_side==0的行，复制对应主驾侧行的have_run,is_pulse_ok,delta_vx(kph),delta_vy(kph),delta_v(kph) 的值
for idx, row in distribution_df.iterrows():
    if row['is_driver_side'] == 0:
        driver_case_id = row['case_id'] - 50000
        driver_row = distribution_df[distribution_df['case_id'] == driver_case_id]
        if not driver_row.empty:
            distribution_df.at[idx, 'have_run'] = driver_row.iloc[0]['have_run']
            distribution_df.at[idx, 'is_pulse_ok'] = driver_row.iloc[0]['is_pulse_ok']
            distribution_df.at[idx, 'delta_vx(kph)'] = driver_row.iloc[0]['delta_vx(kph)']
            distribution_df.at[idx, 'delta_vy(kph)'] = driver_row.iloc[0]['delta_vy(kph)']
            distribution_df.at[idx, 'delta_v(kph)'] = driver_row.iloc[0]['delta_v(kph)']
        else:
            print(f"Warning: Corresponding driver side case_id {driver_case_id} not found for passenger side case_id {row['case_id']}.")
# 保存更新后的distribution文件
if new_distribution_path.endswith('.npz'):
    np.savez(new_distribution_path, **{col: distribution_df[col].values for col in distribution_df.columns})
elif new_distribution_path.endswith('.csv'):
    distribution_df.to_csv(new_distribution_path, index=False)
    print("Final updated distribution file has been saved.")

# %% ex.读取resample.csv，获取需要修改采样值的case_id列表，增加座椅前后位置sp
import pandas as pd
resample_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\resample_before1023.csv'
distribution_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1023_V2.csv'
new_distribution_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1024.csv'
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
# resample只有一列，没有表头
resample_cases = pd.read_csv(resample_path, header=None).iloc[:, 0].dropna().astype(int).to_list()
print(f"Total case_ids to update sp: {len(resample_cases)}")
# 三种体型假人各有范围：5%假人:[+10mm, +110mm]；50%假人: [-80mm, +80mm]; 95%假人: [-110mm, +40mm] 向前移动为正，向后移动为负。
# 先根据三种假人类型，画出resample_cases的各自的sp分布直方图，确认范围合理性
import matplotlib.pyplot as plt
import numpy as np
occupant_types = [1, 2, 3]
for occupant_type in occupant_types:

    sp_values = distribution_df[(distribution_df['occupant_type'] == occupant_type) & (distribution_df['case_id'].isin(resample_cases))]['sp'].dropna().to_list()
    plt.figure()
    plt.hist(sp_values, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.title(f'sp Distribution for occupant_type {occupant_type}')
    plt.xlabel('sp (mm)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

for case_id in resample_cases:
    if case_id in distribution_df.index:

        if distribution_df.at[case_id, 'is_injury_ok'] == True:
            print(f"!Warning: case_id {case_id} has is_injury_ok=True, skipping sp update.")
            continue
        if distribution_df.at[case_id, 'is_pulse_ok'] != True:
            print(f"INFO: case_id {case_id} has is_pulse_ok!=True, skipping sp update.")
            continue
        occupant_type = distribution_df.at[case_id, 'occupant_type']
        old_sp = distribution_df.at[case_id, 'sp']
        if occupant_type == 1:  # 5%假人
            # new_sp = np.clip((old_sp + 110) / 2, 10, 110)
            new_sp = np.random.uniform(40, 85)  
        elif occupant_type == 2:  # 50%假人
            # new_sp = np.clip((old_sp + 80) / 2, -80, 80)
            new_sp = np.random.uniform(-30, 35)  
        elif occupant_type == 3:  # 95%假人
            # new_sp = np.clip((old_sp + 40) / 2, -110, 40)
            new_sp = np.random.uniform(-60, 5)  
        else:
            print(f"Error: Unknown occupant_type {occupant_type} for case_id {case_id}. Skipping sp update.")
            continue
        distribution_df.at[case_id, 'sp'] = new_sp
        # print(f"Updated sp for case_id {case_id} to {new_sp:.2f} mm.")
    else:
        print(f"Error: case_id {case_id} from resample not found in distribution.")
# 画出更新后的resample_cases的各自的sp分布直方图，确认范围合理性
for occupant_type in occupant_types:
    sp_values = distribution_df[(distribution_df['occupant_type'] == occupant_type) & (distribution_df['case_id'].isin(resample_cases))]['sp'].dropna().to_list()
    plt.figure()
    plt.hist(sp_values, bins=20, color='green', edgecolor='black', alpha=0.7)
    plt.title(f'Updated sp Distribution for occupant_type {occupant_type}')
    plt.xlabel('sp (mm)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

# 保存更新后的distribution文件
if new_distribution_path.endswith('.npz'):
    np.savez(new_distribution_path, **{col: distribution_df[col].values for col in distribution_df.columns})
elif new_distribution_path.endswith('.csv'):
    distribution_df.to_csv(new_distribution_path, index=False)
    print("Updated distribution file with new sp values has been saved.")

# %%

# %%