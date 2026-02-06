# -*- coding: utf-8 -*-
'''
采样相关的额外操作或者对distribution文件的操作
'''

# %% 验证和可视化VCS或MADYMO采样结果（20251220-MADYMO采样验证）
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
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
    # 排除is_pulse_ok为False的case_id, 不包含is_pulse_ok为NaN的样本
    if 'case_id' in data and 'is_pulse_ok' in data:
        print("-" * 60)
        print("*排除is_pulse_ok为False的case_id, 但不排除is_pulse_ok为NaN的样本")
        data_df = pd.DataFrame(data)
        initial_count = len(data_df)
        data_df = data_df[~(data_df['is_pulse_ok'] == False)]
        filtered_count = len(data_df)
        data = {col: data_df[col].values for col in data_df.columns}
        print(f"*排除is_pulse_ok为False后，剩余样本数: {filtered_count} (初始样本数: {initial_count})")
        print("-" * 60)
    # 限定主驾侧5th假人（OT=1）的样本进行验证和可视化
    if 'OT' in data and 'is_driver_side' in data:
        print("-" * 60)
        print("*限定主驾侧5th假人（OT=1）的样本进行验证和可视化")
        data_df = pd.DataFrame(data)
        initial_count = len(data_df)
        data_df = data_df[(data_df['OT'] == 1) & (data_df['is_driver_side'] == 1)]
        filtered_count = len(data_df)
        data = {col: data_df[col].values for col in data_df.columns}
        print(f"*限定主驾侧5th假人后，剩余样本数: {filtered_count} (初始样本数: {initial_count})")
        print("-" * 60)
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

    else:  # MADYMO - 新版参数
        # 20251220版约束系统参数
        params_to_check = ['OT', 'LL1', 'LL2', 'BTF', 'LLATTF', 'DZ', 'AFT', 'SP', 'SH', 'RA', 'PTF']
        
        # 全局允许范围（*不*替代基于 OT/is_driver_side 的分段约束，只做快速合理性过滤）
        param_ranges = {
            'LL1': (1.0, 7.0),      # 一级限力值 kN
            'LL2': (0.5, 4.5),      # 二级限力值 kN
            'BTF': (10, 100),       # 预紧器点火时刻 ms
            'LLATTF': (10, 150),    # 二级限力切换时间 ms (最大150表示不切换)
            'AFT': (10, 100),       # 气囊点火时刻 ms
            'SP': (-110, 110),      # 座椅前后位置 mm (完整范围，实际根据体型和主副驾有约束)
            'SH': (-10, 60),       # 座椅高度 mm（全局过滤范围；具体按 OT/is_driver_side 校验）
            'PTF': (17, 107),       # 腰部预紧器点火时间 ms (= BTF + 7)
        }
        
        discrete_params = {
            'OT': [1, 2, 3],                    # 乘员体型: 1=5th, 2=50th, 3=95th
            'DZ': [1, 3, 4],                    # D环高度: 5th->1, 50th->3, 95th->4
            'RA': [15, 20, 25, 30, 35, 40],     # 座椅靠背角度 (主驾15-35, 副驾20-40)
        }
        
        special_checks = [
            'LL2_vs_LL1',       # LL2 < LL1
            'AFT_vs_BTF',       # AFT < BTF + 25
            'PTF_vs_BTF',       # PTF = BTF + 7
            'LLATTF_vs_BTF',    # LLATTF >= BTF 且 LLATTF <= 150
            'DZ_vs_OT',         # DZ与OT的对应关系
            'SP_vs_OT_side',    # SP与体型和主副驾的关系
            'SH_vs_OT_side',    # SH（座椅高度）与体型和主副驾的关系（新增）
            'RA_vs_side',       # RA与主副驾的关系
        ]
        
        # 默认参数对（增加 SP vs SH 的联合可视化）
        if param_pairs is None:
            param_pairs = [
                ('LL1', 'LL2'),
                ('BTF', 'LLATTF'),
                ('BTF', 'AFT'),
                ('BTF', 'PTF'),
                ('OT', 'DZ'),
                ('OT', 'SP'),
                ('OT', 'RA'),
                ('SP', 'SH'),
            ]

    print(f"检查参数组: {params_to_check}")
    print("-" * 60)
    
    # 验证参数
    print("--- 开始数据校验 ---")
    all_checks_passed = True
    verification_results = {}

    # 逐行检查是否有：某一行params_to_check中 空值orNAN 的数量在[1, len(params_to_check)-1]，即存在部分参数缺失
    print("检查参数是否存在部分缺失情况...")
    # 使用第一个存在的参数来确定数据长度
    first_param = next((p for p in params_to_check if p in data), None)
    if first_param is None:
        print(f"错误: 未找到任何待检查的参数")
        return None
    
    for i in range(len(data[first_param])):
        nan_count = 0
        for param in params_to_check:
            if param in data:
                val = data[param][i]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    nan_count += 1
            else:
                nan_count += 1
        if 1 <= nan_count < len(params_to_check):
            print(f"警告: {flag}参数中第{i+1}行数据存在部分参数缺失: {nan_count}个NaN值")
            all_checks_passed = False
    if all_checks_passed:
        print(f"{flag}参数中所有行数据均完整或全部缺失，无部分缺失情况。")
    else:
        print(f"{flag}参数中存在部分缺失情况，请检查数据完整性后重新采样。")
        return None

    # 1. 连续参数范围检查
    def check_continuous(param, min_val, max_val):
        if param not in data:
            print(f"  - 警告: 参数 '{param}' 不存在于数据中")
            return False
        param_data = data[param]
        # 跳过NaN值
        valid_data = param_data[~np.isnan(param_data.astype(float))]
        if len(valid_data) == 0:
            print(f"  - 警告: 参数 '{param}' 全为NaN值")
            return True  # NaN值不算错误
        is_valid = np.all((valid_data >= min_val) & (valid_data <= max_val))
        print(f"  - 检查 '{param}' 范围[{min_val}, {max_val}]: {'通过' if is_valid else '失败!!!!!!!'}")
        if not is_valid:
            invalid_count = np.sum((valid_data < min_val) | (valid_data > max_val))
            print(f"    异常值数量: {invalid_count}/{len(valid_data)}")
            print(f"    异常值范围: [{np.min(valid_data):.4f}, {np.max(valid_data):.4f}]")
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
        param_data = data[param]
        # 跳过NaN值
        valid_data = param_data[~np.isnan(param_data.astype(float))]
        if len(valid_data) == 0:
            print(f"  - 警告: 参数 '{param}' 全为NaN值")
            return True
        is_valid = np.all(np.isin(valid_data, allowed_values))
        print(f"  - 检查 '{param}' 离散值{allowed_values}: {'通过' if is_valid else '失败!!!!!!!'}")
        if not is_valid:
            invalid_values = valid_data[~np.isin(valid_data, allowed_values)]
            print(f"    异常值: {np.unique(invalid_values)}")
        return is_valid
    
    for param, allowed_values in discrete_params.items():
        if param in params_to_check:
            result = check_discrete(param, allowed_values)
            verification_results[param] = result
            all_checks_passed &= result

    # 额外：检查 is_driver_side 只包含 0 或 1（避免意外的其他编码）
    if 'is_driver_side' in data:
        vals = data['is_driver_side'][~np.isnan(data['is_driver_side'].astype(float))]
        if len(vals) > 0:
            is_driver_ok = np.all(np.isin(vals.astype(int), [0, 1]))
            print(f"  - 检查 'is_driver_side' 取值仅为 0/1: {'通过' if is_driver_ok else '失败!!!!!!!'}")
            if not is_driver_ok:
                print(f"    非法取值示例: {np.unique(vals[~np.isin(vals.astype(int), [0,1])])}")
            verification_results['is_driver_side'] = is_driver_ok
            all_checks_passed &= is_driver_ok

    # 3. 特殊参数检查
    if flag == 'VCS':
        # 检查重叠率特殊区间: (-1, -0.25]∪[0.25, 1]
        if 'overlap' in data:
            overlap_data = data['overlap'][~np.isnan(data['overlap'])]
            if len(overlap_data) > 0:
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
                is_relation_valid = np.all(
                    (np.abs(angles_to_check) >= 30) & 
                    (np.sign(angles_to_check) != np.sign(overlaps_to_check))
                )
                print(f"  - 检查重叠率绝对值在0.25~0.3之间的样本的 'impact_angle' (与重叠率异号且绝对值>=30度): {'通过' if is_relation_valid else '失败!!!!!!!'}")
                if not is_relation_valid:
                    failed_mask = ~((np.abs(angles_to_check) >= 30) & (np.sign(angles_to_check) != np.sign(overlaps_to_check)))
                    num_failed = np.sum(failed_mask)
                    print(f"    共有 {num_failed} 个样本不满足此项关联检查。")
                verification_results['overlap_angle_relation'] = is_relation_valid
                all_checks_passed &= is_relation_valid
            else:
                print("  - ! warning ! 无重叠率绝对值在0.25~0.3之间的样本，注意采样范围。")
                verification_results['overlap_angle_relation'] = True

    elif flag == 'MADYMO':
        # ==================== 新版MADYMO特殊检查 ====================
        
        # 检查 LL2 < LL1
        if 'LL2_vs_LL1' in special_checks and 'LL1' in data and 'LL2' in data:
            ll1_data = data['LL1']
            ll2_data = data['LL2']
            valid_mask = ~(np.isnan(ll1_data) | np.isnan(ll2_data))
            if np.any(valid_mask):
                ll1_valid = ll1_data[valid_mask]
                ll2_valid = ll2_data[valid_mask]
                is_ll2_valid = np.all(ll2_valid < ll1_valid)
                print(f"  - 检查 'LL2' < 'LL1': {'通过' if is_ll2_valid else '失败!!!!!!!'}")
                if not is_ll2_valid:
                    failed_count = np.sum(ll2_valid >= ll1_valid)
                    print(f"    不满足条件的样本数: {failed_count}/{len(ll1_valid)}")
                verification_results['LL2_vs_LL1'] = is_ll2_valid
                all_checks_passed &= is_ll2_valid
        
        # 检查 AFT < BTF + 25
        if 'AFT_vs_BTF' in special_checks and 'AFT' in data and 'BTF' in data:
            aft_data = data['AFT']
            btf_data = data['BTF']
            valid_mask = ~(np.isnan(aft_data) | np.isnan(btf_data))
            if np.any(valid_mask):
                aft_valid = aft_data[valid_mask]
                btf_valid = btf_data[valid_mask]
                is_aft_valid = np.all(aft_valid < (btf_valid + 25))
                print(f"  - 检查 'AFT' < 'BTF' + 25: {'通过' if is_aft_valid else '失败!!!!!!!'}")
                if not is_aft_valid:
                    failed_count = np.sum(aft_valid >= (btf_valid + 25))
                    print(f"    不满足条件的样本数: {failed_count}/{len(aft_valid)}")
                verification_results['AFT_vs_BTF'] = is_aft_valid
                all_checks_passed &= is_aft_valid
        
        # 检查 PTF = BTF + 7
        if 'PTF_vs_BTF' in special_checks and 'PTF' in data and 'BTF' in data:
            ptf_data = data['PTF']
            btf_data = data['BTF']
            valid_mask = ~(np.isnan(ptf_data) | np.isnan(btf_data))
            if np.any(valid_mask):
                ptf_valid = ptf_data[valid_mask]
                btf_valid = btf_data[valid_mask]
                is_ptf_valid = np.allclose(ptf_valid, btf_valid + 7.0, rtol=1e-5)
                print(f"  - 检查 'PTF' = 'BTF' + 7: {'通过' if is_ptf_valid else '失败!!!!!!!'}")
                if not is_ptf_valid:
                    diff = np.abs(ptf_valid - (btf_valid + 7.0))
                    failed_count = np.sum(diff > 0.01)
                    print(f"    不满足条件的样本数: {failed_count}/{len(ptf_valid)}")
                verification_results['PTF_vs_BTF'] = is_ptf_valid
                all_checks_passed &= is_ptf_valid
        
        # 检查 LLATTF >= BTF 且 LLATTF <= 150
        if 'LLATTF_vs_BTF' in special_checks and 'LLATTF' in data and 'BTF' in data:
            llattf_data = data['LLATTF']
            btf_data = data['BTF']
            valid_mask = ~(np.isnan(llattf_data) | np.isnan(btf_data))
            if np.any(valid_mask):
                llattf_valid = llattf_data[valid_mask]
                btf_valid = btf_data[valid_mask]
                # LLATTF应该在[BTF, 150]范围内
                is_llattf_valid = np.all((llattf_valid >= btf_valid) & (llattf_valid <= 150))
                print(f"  - 检查 'LLATTF' 在 [BTF, 150] 范围内: {'通过' if is_llattf_valid else '失败!!!!!!!'}")
                if not is_llattf_valid:
                    failed_mask = (llattf_valid < btf_valid) | (llattf_valid > 150)
                    failed_count = np.sum(failed_mask)
                    print(f"    不满足条件的样本数: {failed_count}/{len(llattf_valid)}")
                verification_results['LLATTF_vs_BTF'] = is_llattf_valid
                all_checks_passed &= is_llattf_valid
        
        # 检查 DZ 与 OT 的对应关系: OT=1->DZ=1, OT=2->DZ=3, OT=3->DZ=4
        if 'DZ_vs_OT' in special_checks and 'DZ' in data and 'OT' in data:
            dz_data = data['DZ']
            ot_data = data['OT']
            valid_mask = ~(np.isnan(dz_data) | np.isnan(ot_data))
            if np.any(valid_mask):
                dz_valid = dz_data[valid_mask]
                ot_valid = ot_data[valid_mask]
                
                # 定义正确的对应关系
                expected_dz = np.where(ot_valid == 1, 1, np.where(ot_valid == 2, 3, 4))
                is_dz_valid = np.all(dz_valid == expected_dz)
                print(f"  - 检查 'DZ' 与 'OT' 对应关系 (OT=1->DZ=1, OT=2->DZ=3, OT=3->DZ=4): {'通过' if is_dz_valid else '失败!!!!!!!'}")
                if not is_dz_valid:
                    failed_count = np.sum(dz_valid != expected_dz)
                    print(f"    不满足条件的样本数: {failed_count}/{len(dz_valid)}")
                verification_results['DZ_vs_OT'] = is_dz_valid
                all_checks_passed &= is_dz_valid
        
        # 检查 SP 与体型和主副驾的关系
        if 'SP_vs_OT_side' in special_checks and 'SP' in data and 'OT' in data and 'is_driver_side' in data:
            sp_data = data['SP']
            ot_data = data['OT']
            side_data = data['is_driver_side']
            valid_mask = ~(np.isnan(sp_data) | np.isnan(ot_data) | np.isnan(side_data))
            
            if np.any(valid_mask):
                sp_valid = sp_data[valid_mask]
                ot_valid = ot_data[valid_mask]
                side_valid = side_data[valid_mask]
                
                # SP范围定义
                # 主驾 (is_driver_side=1): 5th: [+40, +110], 50th: [-40, +60], 95th: [-110, +20]
                # 副驾 (is_driver_side=0): 5th/50th: [-110, +110], 95th: [-110, +49]
                is_sp_valid = True
                
                # 主驾 5th
                mask_driver_5th = (side_valid == 1) & (ot_valid == 1)
                if np.any(mask_driver_5th):
                    is_sp_valid &= np.all((sp_valid[mask_driver_5th] >= 40) & (sp_valid[mask_driver_5th] <= 110))
                
                # 主驾 50th
                mask_driver_50th = (side_valid == 1) & (ot_valid == 2)
                if np.any(mask_driver_50th):
                    is_sp_valid &= np.all((sp_valid[mask_driver_50th] >= -40) & (sp_valid[mask_driver_50th] <= 60)) # 20260114修改主驾50th范围
                
                # 主驾 95th
                mask_driver_95th = (side_valid == 1) & (ot_valid == 3)
                if np.any(mask_driver_95th):
                    is_sp_valid &= np.all((sp_valid[mask_driver_95th] >= -110) & (sp_valid[mask_driver_95th] <= 20))
                
                # 副驾 5th/50th
                mask_pass_5th_50th = (side_valid == 0) & ((ot_valid == 1) | (ot_valid == 2))
                if np.any(mask_pass_5th_50th):
                    is_sp_valid &= np.all((sp_valid[mask_pass_5th_50th] >= -110) & (sp_valid[mask_pass_5th_50th] <= 110))
                
                # 副驾 95th
                mask_pass_95th = (side_valid == 0) & (ot_valid == 3)
                if np.any(mask_pass_95th):
                    is_sp_valid &= np.all((sp_valid[mask_pass_95th] >= -110) & (sp_valid[mask_pass_95th] <= 49))
                
                print(f"  - 检查 'SP' 与体型和主副驾的对应关系: {'通过' if is_sp_valid else '失败!!!!!!!'}")
                verification_results['SP_vs_OT_side'] = is_sp_valid
                all_checks_passed &= is_sp_valid

                # -------------------- 额外严格校验：(SP, SH) 必须位于采样端定义的多边形内（逐组） --------------------
                try:
                    # 与采样端一致的多边形顶点（单位：mm）
                    SEAT_POLYS = {
                        (1, 1): [(40, 0), (110, 5.5), (110, 60), (80, 60), (80, 30), (40, 30)],
                        (1, 2): [(-40, -2.2), (60, 3.3), (60, 60), (-40, 60)],
                        (1, 3): [(-110, -10), (20, -10), (20, 70), (-110, 70)],
                        (0, 1): [(-110, -10), (110, -10), (110, 70), (-110, 70)],
                        (0, 2): [(-110, -10), (110, -10), (110, 70), (-110, 70)],
                        (0, 3): [(-110, -10), (49, -10), (49, 70), (-110, 70)],
                    }

                    sp_sh_mask = ~(np.isnan(data.get('SP', np.array([np.nan]*len(data[first_param]))).astype(float)) |
                                    np.isnan(data.get('SH', np.array([np.nan]*len(data[first_param]))).astype(float)) |
                                    np.isnan(data.get('OT', np.array([np.nan]*len(data[first_param]))).astype(float)) |
                                    np.isnan(data.get('is_driver_side', np.array([np.nan]*len(data[first_param]))).astype(float)))

                    if np.any(sp_sh_mask):
                        sp_all = data['SP'][sp_sh_mask].astype(float)
                        sh_all = data['SH'][sp_sh_mask].astype(float)
                        ot_all = data['OT'][sp_sh_mask].astype(int)
                        side_all = data['is_driver_side'][sp_sh_mask].astype(int)

                        outside_count = 0
                        examples = []
                        for (side_val, ot_val), poly in SEAT_POLYS.items():
                            sel = (side_all == side_val) & (ot_all == ot_val)
                            if not np.any(sel):
                                continue
                            pts = np.column_stack((sp_all[sel], sh_all[sel]))
                            path = Path(poly)
                            contains = path.contains_points(pts)
                            if not np.all(contains):
                                n_bad = np.sum(~contains)
                                outside_count += int(n_bad)
                                # collect up to 3 example points
                                bad_idx = np.where(sel)[0][~contains]
                                for bi in bad_idx[:3]:
                                    examples.append((int(side_val), int(ot_val), float(pts[bi,0]), float(pts[bi,1])))

                        is_polygon_ok = (outside_count == 0)
                        print(f"  - 检查 (SP,SH) 是否位于允许多边形内: {'通过' if is_polygon_ok else '失败!!!!!!!'}")
                        if not is_polygon_ok:
                            print(f"    多边形外样本数: {outside_count}; 示例 (side,OT,SP,SH): {examples[:3]}")
                        verification_results['SP_SH_polygon'] = is_polygon_ok
                        all_checks_passed &= is_polygon_ok
                except Exception as _e:
                    print(f"  - 多边形包含检查失败（跳过）: {_e}")

        # -------------------- 新增：SH（座椅高度）按 OT 与主/副驾的分段范围校验 --------------------
        if 'SH_vs_OT_side' in special_checks and 'SH' in data and 'OT' in data and 'is_driver_side' in data:
            sh_data = data['SH']
            ot_data = data['OT']
            side_data = data['is_driver_side']
            valid_mask = ~(np.isnan(sh_data) | np.isnan(ot_data) | np.isnan(side_data))
            
            if np.any(valid_mask):
                sh_valid = sh_data[valid_mask]
                ot_valid = ot_data[valid_mask].astype(int)
                side_valid = side_data[valid_mask].astype(int)

                # 从采样端 SEAT_CONSTRAINTS 推断的每组 (is_driver_side, OT) 的 SH 最小/最大值（mm）
                SH_RANGES = {
                    (1, 1): (0.0, 60.0),    # 主驾 5th
                    (1, 2): (-3.0, 60.0),   # 主驾 50th
                    (1, 3): (-10.0, 70.0),  # 主驾 95th
                    (0, 1): (-10.0, 70.0),  # 副驾 5th
                    (0, 2): (-10.0, 70.0),  # 副驾 50th
                    (0, 3): (-10.0, 70.0),  # 副驾 95th
                }

                is_sh_valid = True
                failed_details = []
                for (side_val, ot_val), (sh_min, sh_max) in SH_RANGES.items():
                    mask = (side_valid == side_val) & (ot_valid == ot_val)
                    if np.any(mask):
                        block = sh_valid[mask]
                        ok = np.all((block >= sh_min) & (block <= sh_max))
                        is_sh_valid &= ok
                        if not ok:
                            failed_count = np.sum((block < sh_min) | (block > sh_max))
                            failed_details.append(((side_val, ot_val), failed_count, sh_min, sh_max))

                print(f"  - 检查 'SH' 与体型和主副驾的对应关系: {'通过' if is_sh_valid else '失败!!!!!!!'}")
                if not is_sh_valid:
                    for (side_val, ot_val), cnt, smin, smax in failed_details:
                        side_s = '主驾' if side_val == 1 else '副驾'
                        print(f"    不满足 ({side_s}, OT={ot_val}) 的样本数: {cnt}，允许范围: [{smin}, {smax}] mm")

                verification_results['SH_vs_OT_side'] = is_sh_valid
                all_checks_passed &= is_sh_valid
        
        # 检查 RA 与主副驾的关系
        if 'RA_vs_side' in special_checks and 'RA' in data and 'is_driver_side' in data:
            ra_data = data['RA']
            side_data = data['is_driver_side']
            valid_mask = ~(np.isnan(ra_data) | np.isnan(side_data))
            
            if np.any(valid_mask):
                ra_valid = ra_data[valid_mask]
                side_valid = side_data[valid_mask]
                
                # RA离散值：主驾 [15, 20, 25, 30]°，副驾 [20, 25, 30, 35, 40]°
                is_ra_valid = True
                
                # 主驾
                mask_driver = (side_valid == 1)
                if np.any(mask_driver):
                    driver_ra_allowed = [15, 20, 25, 30] # 20260114修改主驾最大35为30
                    is_ra_valid &= np.all(np.isin(ra_valid[mask_driver], driver_ra_allowed))
                
                # 副驾
                mask_pass = (side_valid == 0)
                if np.any(mask_pass):
                    pass_ra_allowed = [20, 25, 30, 35, 40]
                    is_ra_valid &= np.all(np.isin(ra_valid[mask_pass], pass_ra_allowed))
                
                print(f"  - 检查 'RA' 与主副驾的对应关系 (主驾[15-35], 副驾[20-40]): {'通过' if is_ra_valid else '失败!!!!!!!'}")
                if not is_ra_valid:
                    # 输出详细信息
                    if np.any(mask_driver):
                        driver_ra = ra_valid[mask_driver]
                        invalid_driver = driver_ra[~np.isin(driver_ra, [15, 20, 25, 30, 35])]
                        if len(invalid_driver) > 0:
                            print(f"    主驾异常值: {np.unique(invalid_driver)}")
                    if np.any(mask_pass):
                        pass_ra = ra_valid[mask_pass]
                        invalid_pass = pass_ra[~np.isin(pass_ra, [20, 25, 30, 35, 40])]
                        if len(invalid_pass) > 0:
                            print(f"    副驾异常值: {np.unique(invalid_pass)}")
                verification_results['RA_vs_side'] = is_ra_valid
                all_checks_passed &= is_ra_valid
    
    print(f"\n--- 校验总结: {'所有检查均已通过！' if all_checks_passed else '存在未通过的检查项！'} ---\n")
      
    print("--- 开始生成可视化图表 ---")
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ==================== 额外统计：关键区间样本比例 ====================
    if flag == 'MADYMO':
        print("\n--- 关键区间样本比例统计 ---")
        
        # LL1关键区间统计：[1.5, 3.0] kN
        if 'LL1' in data:
            ll1_data = data['LL1'][~np.isnan(data['LL1'].astype(float))]
            if len(ll1_data) > 0:
                ll1_in_key_range = np.sum((ll1_data >= 1.5) & (ll1_data <= 3.0))
                ll1_key_ratio = ll1_in_key_range / len(ll1_data) * 100
                print(f"  LL1 在 [1.5, 3.0] kN 区间的样本比例: {ll1_key_ratio:.2f}% ({ll1_in_key_range}/{len(ll1_data)})")
        
        # LL2关键区间统计：[0.5, 1.5] kN
        if 'LL2' in data:
            ll2_data = data['LL2'][~np.isnan(data['LL2'].astype(float))]
            if len(ll2_data) > 0:
                ll2_in_key_range = np.sum((ll2_data >= 0.5) & (ll2_data <= 1.5))
                ll2_key_ratio = ll2_in_key_range / len(ll2_data) * 100
                print(f"  LL2 在 [0.5, 1.5] kN 区间的样本比例: {ll2_key_ratio:.2f}% ({ll2_in_key_range}/{len(ll2_data)})")
        
        # LLATTF=150ms统计（代表不切换二级限力）
        if 'LLATTF' in data:
            llattf_data = data['LLATTF'][~np.isnan(data['LLATTF'].astype(float))]
            if len(llattf_data) > 0:
                llattf_150_count = np.sum(llattf_data == 150)
                llattf_150_ratio = llattf_150_count / len(llattf_data) * 100
                print(f"  LLATTF = 150 ms (不切换二级限力) 的样本比例: {llattf_150_ratio:.2f}% ({llattf_150_count}/{len(llattf_data)})")
        
        print("-" * 60)
    
    # 3.1 各参数的一维分布图
    print("正在生成一维分布图...")
    available_params = [p for p in params_to_check if p in data and not np.all(np.isnan(data[p].astype(float)))]
    
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

        for i, param in enumerate(available_params):
            param_data = data[param][~np.isnan(data[param].astype(float))]
            if len(param_data) > 0:
                if param in discrete_params:
                    # 离散参数使用条形图
                    unique_vals, counts = np.unique(param_data, return_counts=True)
                    axes1[i].bar(unique_vals.astype(str), counts / len(param_data), color='blue', edgecolor='black')
                else:
                    # 连续参数使用直方图
                    bins = 20
                    if flag == 'VCS':
                        if param == 'impact_velocity':
                            bins = np.arange(25, 70, 5)
                        elif param == 'impact_angle':
                            bins = np.arange(-45, 50, 5)
                        elif param == 'overlap':
                            bins = np.arange(-1.0, 1.1, 0.1)
                    elif flag == 'MADYMO':
                        if param == 'LL1':
                            bins = np.arange(1.0, 7.5, 0.5)
                        elif param == 'LL2':
                            bins = np.arange(0.5, 5.0, 0.5)
                        elif param in ['BTF', 'AFT', 'LLATTF', 'PTF']:
                            bins = np.arange(10, 160, 10)
                        elif param == 'SP':
                            bins = np.arange(-120, 130, 20)
                    sns.histplot(param_data, kde=True, ax=axes1[i], stat="density", bins=bins, color='blue', edgecolor='black')
                
                axes1[i].set_title(f'{param}分布')
                axes1[i].set_xlabel('值')
                axes1[i].set_ylabel('频率')
                
                # ==================== 在图中标注关键区间比例 ====================
                if flag == 'MADYMO':
                    if param == 'LL1':
                        # 标注 [1.5, 3.0] kN 区间比例
                        ll1_in_range = np.sum((param_data >= 1.5) & (param_data <= 3.0))
                        ll1_ratio = ll1_in_range / len(param_data) * 100
                        # 绘制关键区间背景
                        axes1[i].axvspan(1.5, 3.0, alpha=0.2, color='red', label=f'[1.5,3.0]kN: {ll1_ratio:.1f}%')
                        axes1[i].legend(loc='upper right', fontsize=8)
                    
                    elif param == 'LL2':
                        # 标注 [0.5, 1.5] kN 区间比例
                        ll2_in_range = np.sum((param_data >= 0.5) & (param_data <= 1.5))
                        ll2_ratio = ll2_in_range / len(param_data) * 100
                        # 绘制关键区间背景
                        axes1[i].axvspan(0.5, 1.5, alpha=0.2, color='red', label=f'[0.5,1.5]kN: {ll2_ratio:.1f}%')
                        axes1[i].legend(loc='upper right', fontsize=8)
                    
                    elif param == 'LLATTF':
                        # 标注 LLATTF=150ms 的比例
                        llattf_150 = np.sum(param_data == 150)
                        llattf_150_ratio = llattf_150 / len(param_data) * 100
                        # 在150ms处绘制垂直线并标注
                        axes1[i].axvline(x=150, color='red', linestyle='--', linewidth=2, label=f'150ms(不切换): {llattf_150_ratio:.1f}%')
                        axes1[i].legend(loc='upper right', fontsize=8)

                # -------------------- 新增：SP/SH/RA 按主/副驾与按 OT 的分组分布图 --------------------
                if flag == 'MADYMO' and param in ('SP', 'SH', 'RA'):
                    try:
                        df_plot = pd.DataFrame({
                            param: data[param].astype(float),
                            'is_driver_side': data.get('is_driver_side', np.full(len(data[param]), np.nan)).astype(float),
                            'OT': data.get('OT', np.full(len(data[param]), np.nan)).astype(float)
                        })
                        df_plot = df_plot.dropna(subset=[param])

                        # 按主/副驾叠加（RA 为离散，使用 countplot）
                        fig_a, ax_a = plt.subplots(1, 1, figsize=(6, 4))
                        if param == 'RA':
                            sns.countplot(x=param, hue='is_driver_side', data=df_plot, ax=ax_a, palette='Set2')
                            ax_a.set_ylabel('count')
                        else:
                            sns.histplot(data=df_plot, x=param, hue='is_driver_side', kde=False, stat='density', bins=20, ax=ax_a, palette='Set2', alpha=0.6)
                            ax_a.set_ylabel('density')
                        ax_a.set_title(f'{param} distributions by side')
                        side_filename = os.path.join(output_dir, f'{flag}_{param}_by_side.png')
                        fig_a.tight_layout()
                        fig_a.savefig(side_filename, dpi=300, bbox_inches='tight')
                        plt.close(fig_a)

                        # 按 OT 分面显示（若 OT 存在且非全 NaN）
                        if not df_plot['OT'].isna().all():
                            g = sns.FacetGrid(df_plot, col='OT', col_wrap=3, height=3.2, sharex=True, sharey=False)
                            if param == 'RA':
                                g.map(sns.countplot, param, order=sorted(df_plot[param].unique()))
                            else:
                                g.map(sns.histplot, param, kde=False, bins=20)
                            ot_filename = os.path.join(output_dir, f'{flag}_{param}_by_OT.png')
                            plt.subplots_adjust(top=0.9)
                            g.fig.suptitle(f'{param} distributions by OT')
                            g.fig.savefig(ot_filename, dpi=300, bbox_inches='tight')
                            plt.close(g.fig)

                        print(f"  - 已保存按主/副驾与按OT分组的 {param} 分布图: {side_filename}, {ot_filename if 'ot_filename' in locals() else '(no OT plot)'}")
                    except Exception as e:
                        print(f"  - 绘制 {param} 分组分布图失败: {e}")
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
                      and not np.all(np.isnan(data[p1].astype(float))) 
                      and not np.all(np.isnan(data[p2].astype(float)))]
    
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
            data1 = data[param1].astype(float)
            data2 = data[param2].astype(float)
            valid_mask = ~(np.isnan(data1) | np.isnan(data2))
            if np.any(valid_mask):
                axes2[i].scatter(data1[valid_mask], data2[valid_mask], alpha=0.6, s=20)
                axes2[i].set_title(f'{param1} vs {param2}')
                axes2[i].set_xlabel(param1)
                axes2[i].set_ylabel(param2)
                axes2[i].grid(True, alpha=0.3)

                # 如果是 SP vs SH 且为 MADYMO：
                # - 仅绘制数据中实际出现的 (is_driver_side, OT) 对应的多边形，避免过度叠加造成视觉干扰
                # - 在多边形顶点绘制小点并标注顶点序号（简短坐标可选）以便人工核查
                if flag == 'MADYMO' and ((param1 == 'SP' and param2 == 'SH') or (param1 == 'SH' and param2 == 'SP')):
                    try:
                        SEAT_POLYS = {
                            (1, 1): [(40, 0), (110, 5.5), (110, 60), (80, 60), (80, 30), (40, 30)],
                            (1, 2): [(-40, -2.2), (60, 3.3), (60, 60), (-40, 60)],
                            (1, 3): [(-110, -10), (20, -10), (20, 70), (-110, 70)],
                            (0, 1): [(-110, -10), (110, -10), (110, 70), (-110, 70)],
                            (0, 2): [(-110, -10), (110, -10), (110, 70), (-110, 70)],
                            (0, 3): [(-110, -10), (49, -10), (49, 70), (-110, 70)],
                        }

                        # 获取当前绘图中实际存在的 (side, OT) 组合 —— 仅绘制这些多边形
                        side_arr = data.get('is_driver_side', np.full(len(data[first_param]), np.nan)).astype(float)
                        ot_arr = data.get('OT', np.full(len(data[first_param]), np.nan)).astype(float)
                        sp_arr = data.get('SP', np.full(len(data[first_param]), np.nan)).astype(float)
                        sh_arr = data.get('SH', np.full(len(data[first_param]), np.nan)).astype(float)

                        mask_pairs = valid_mask.copy() if 'valid_mask' in locals() else (~np.isnan(sp_arr) & ~np.isnan(sh_arr))
                        present_pairs = set()
                        if np.any(mask_pairs):
                            side_sel = side_arr[mask_pairs].astype(int)
                            ot_sel = ot_arr[mask_pairs].astype(int)
                            for s, o in zip(side_sel, ot_sel):
                                if (int(s), int(o)) in SEAT_POLYS:
                                    present_pairs.add((int(s), int(o)))

                        # If no specific pairs present, draw only the most-relevant polygon (OT where data exists) or skip
                        if len(present_pairs) == 0:
                            # try to infer OT from data values near the plotted region
                            uniq_ot = np.unique(ot_arr[~np.isnan(ot_arr)])
                            for otv in uniq_ot:
                                for sv in (1, 0):
                                    if (sv, int(otv)) in SEAT_POLYS:
                                        present_pairs.add((sv, int(otv)))
                                        break

                        colors = plt.cm.Set2.colors
                        color_map = {}
                        for idx, pair in enumerate(sorted(present_pairs)):
                            color_map[pair] = colors[idx % len(colors)]

                        for pair, poly in SEAT_POLYS.items():
                            if pair not in present_pairs:
                                continue
                            poly = np.array(poly)
                            col = color_map.get(pair, 'C3')
                            # 多边形边界
                            axes2[i].plot(np.append(poly[:,0], poly[0,0]), np.append(poly[:,1], poly[0,1]),
                                          linestyle='--', color=col, linewidth=1.2, alpha=0.9,
                                          label=f'side={pair[0]},OT={pair[1]}')
                            # 顶点标注（index + 简短坐标）
                            for vi, (vx, vy) in enumerate(poly):
                                axes2[i].scatter(vx, vy, marker='o', s=18, color=col, edgecolor='black', zorder=5)
                                axes2[i].text(vx, vy, f'{vi}', fontsize=7, color=col, va='bottom', ha='right', zorder=6)

                        # 图例（仅当至少绘制了 1 个多边形时）
                        if len(present_pairs) > 0:
                            axes2[i].legend(title='Seat poly (side,OT)', fontsize=8, title_fontsize=9, loc='upper left')

                        print(f"  - 绘制 seat-polygons（仅针对数据中存在的组合）: {sorted(present_pairs)}")
                    except Exception as _e:
                        print(f"  - 绘制 seat-polygons 失败: {_e}")
        
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
    print("-" * 60 + "\n")
    
    return verification_results


if __name__ == '__main__':
    # # VCS验证
    # verify_and_visualize_params(
    #     r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1220.csv',
    #     flag='VCS',
    #     output_dir='VCS_sample_verification_1220',
    #     param_pairs=[('impact_velocity', 'impact_angle'), ('impact_velocity', 'overlap'), ('impact_angle', 'overlap')]
    # )
    
    # MADYMO验证
    verify_and_visualize_params(
        r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0206.csv',
        flag='MADYMO',
        output_dir='MADYMO_sample_verification_0206',
        param_pairs=[
            ('LL1', 'LL2'),
            ('BTF', 'LLATTF'),
            ('BTF', 'AFT'),
            ('BTF', 'PTF'),
            ('OT', 'DZ'),
            ('OT', 'SP'),
            ('OT', 'RA'),
            ('is_driver_side', 'SP'),
            ('is_driver_side', 'SH'),
            ('is_driver_side', 'RA'),
            ('SP', 'SH')
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


# %% -Final.将头颈胸损伤标签（HIC15, Dmax, Nij）添加到distribution文件中
import numpy as np
import pandas as pd
distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0130.csv'
new_distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0205.csv'
Injury_labels_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\Injury_labels_0205.xlsx'
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
    

# %% 将指定case_id的损伤标签改为False 或 NaN
import numpy as np
import pandas as pd
distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0205.csv'
new_distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0205_.csv'
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
# case_ids_to_update = [52145, 54284, 53844, 53448, 53641, 56952, 50813, 54254, 57034, 53690, 55617, 51667, 51970, 53044, 54957, 53176, 56323, 54196, 54921, 55707]
case_ids_to_update = [7865]
for case_id in case_ids_to_update:
    if case_id in distribution_df.index:
        distribution_df.at[case_id, 'is_injury_ok'] = False  # nan or False
        # distribution_df.at[case_id, 'HIC15'] = np.nan
        # distribution_df.at[case_id, 'Dmax'] = np.nan
        # distribution_df.at[case_id, 'Nij'] = np.nan
        print(f"Updated injury labels to False/NaN for case_id {case_id}.")
    else:
        print(f"Warning: case_id {case_id} not found in distribution.")
# 保存更新后的distribution文件
if new_distribution_path.endswith('.npz'):
    np.savez(new_distribution_path, **{col: distribution_df[col].values for col in distribution_df.columns})
elif new_distribution_path.endswith('.csv'):
    distribution_df.to_csv(new_distribution_path, index=False)
    print("Updated distribution file with injury labels has been saved.")
# %%
# %%