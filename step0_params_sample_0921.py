# %% 第一部分：对三个碰撞工况参数进行采样
import numpy as np
import pandas as pd
from scipy.stats import qmc
import matplotlib.pyplot as plt
import seaborn as sns

def sample_collision_params(n_samples=6000, skip_points=1024, method='uniform', 
                            filename='distribution.npz', case_ids=None, seed=20252025):
    """
    对碰撞工况参数进行采样
    
    参数:
    - n_samples: 采样数量
    - skip_points: 跳过的初始点数量
    - method: 'uniform'或'non_uniform'，采样方法
    - filename: '.npz'或'.csv' 结尾的输出文件名
    - case_ids: 指定的case_id列，如果为None则自动生成
    - seed: 随机种子
    
    返回:
    - 文件名
    """
    print(f"开始对碰撞工况参数进行{method}采样...")
    print(f"  - 采样数量: {n_samples}")
    print(f"  - 跳过初始点: {skip_points}")
    print(f"  - 随机种子: {seed}")
    
    # 初始化Sobol序列生成器
    sobol = qmc.Sobol(d=3, scramble=True, seed=seed)
    
    # 跳过部分初始点
    sobol.fast_forward(skip_points)
    
    # 生成 [0, 1) 范围内的均匀Sobol样本
    uniform_samples = sobol.random(n=n_samples)
    
    # 根据选择的方法对样本进行转换
    if method == 'uniform':
        # 均匀采样
        impact_velocity = 25.0 + (65.0 - 25.0) * uniform_samples[:, 0] # 碰撞速度 ('impact_velocity') 数据, 分布在 [25, 65] km/h
        impact_angle = -45.0 + (45.0 - (-45.0)) * uniform_samples[:, 1] # 碰撞角度 ('impact_angle') 数据, 分布在 [-45, 45]° 范围内
        
        # 特殊处理重叠率: (-1, -0.25]∪[0.25, 1]
        # 将[0,1)映射到两个区间的联合：(-1, -0.25]∪[0.25, 1]
        u = uniform_samples[:, 2]  # [0, 1)范围的均匀样本

        # 计算两个区间的长度
        interval1_length = -0.25 - (-1.0)  # 0.75
        interval2_length = 1.0 - 0.25      # 0.75
        total_length = interval1_length + interval2_length  # 1.5

        # 将[0,1)按比例分配到两个区间
        threshold = interval1_length / total_length  # ≈0.5

        # 对于u < threshold的样本，映射到(-1, -0.25]
        # 对于u >= threshold的样本，映射到[0.25, 1]
        overlap = np.where(
            u < threshold,
            -1.0 + (u / threshold) * interval1_length,  # 映射到(-1, -0.25]
            0.25 + ((u - threshold) / (1 - threshold)) * interval2_length  # 映射到[0.25, 1]
        )
                
    else:  # 非均匀采样
        # 碰撞速度 ('impact_velocity') 数据, 分布在 [25, 65] km/h
        velocity_histogram_data = [
            [25, 30, 9.0], # 9.0
            [30, 35, 11.0], # 20.0
            [35, 40, 12.0], # 32.0
            [40, 45, 13.5], # 45.5
            [45, 50, 13.5], # 59.0
            [50, 55, 14.0], # 73.0
            [55, 60, 14.0], # 87.0
            [60, 65, 13.0], # 100.0
        ]

        # 碰撞角度 ('impact_angle') 数据, 筛选在 [-45, 45] 度范围内
        angle_histogram_data = [
            [-45, -35, 1.5], 
            [-35, -30, 2.0],
            [-30, -25, 2.5], 
            [-25, -20, 3.0],
            [-20, -15, 4.0],
            [-15, -10, 5.0],
            [-10, -5, 8.0], 
            [-5, 0, 23.0], 

            [0, 5, 23.0],
            [5, 10, 8.0], 
            [10, 15, 5.0],
            [15, 20, 4.0],
            [20, 25, 3.0],
            [25, 30, 2.5],
            [30, 35, 2.0],
            [35, 45, 1.5],
        ]

        # 重叠率 ('overlap') 数据, 调整为 (-1, -0.15]∪[0.15, 1] 范围
        overlap_histogram_data = [
            [-1.0, -0.9, 11.5], 
            [-0.9, -0.8, 8.5], 
            [-0.8, -0.7, 7.0],
            [-0.7, -0.6, 6.0], 
            [-0.6, -0.5, 5.0], 
            [-0.5, -0.4, 4.0],
            [-0.4, -0.3, 3.5], 
            [-0.3, -0.25, 2.0], 

            [0.25, 0.3, 2.0], 
            [0.3, 0.4, 3.5], 
            [0.4, 0.5, 4.5],
            [0.5, 0.6, 5.5], 
            [0.6, 0.7, 6.5], 
            [0.7, 0.8, 7.5],
            [0.8, 0.9, 9.0], 
            [0.9, 1.0, 13.0],
        ]
        
        
        # 实现用于非均匀采样的分段采样器
        def create_piecewise_sampler(histogram_data):
            bins = np.array(histogram_data)
            x_mins, x_maxs, heights = bins[:, 0], bins[:, 1], bins[:, 2]
            
            widths = x_maxs - x_mins
            areas = widths * heights
            bin_probabilities = areas / np.sum(areas)
            cumulative_probabilities = np.cumsum(bin_probabilities)
            cumulative_probabilities[-1] = 1.0
            
            def sampler(u):
                u = np.asarray(u)
                bin_indices = np.searchsorted(cumulative_probabilities, u)
                chosen_x_mins = x_mins[bin_indices]
                chosen_x_maxs = x_maxs[bin_indices]
                chosen_bin_probs = bin_probabilities[bin_indices]
                prev_cum_probs = np.concatenate(([0], cumulative_probabilities[:-1]))
                chosen_prev_cum_probs = prev_cum_probs[bin_indices]
                v = (u - chosen_prev_cum_probs) / chosen_bin_probs
                sampled_values = chosen_x_mins + v * (chosen_x_maxs - chosen_x_mins)
                return sampled_values
            return sampler

        # 创建采样器
        velocity_sampler = create_piecewise_sampler(velocity_histogram_data)
        angle_sampler = create_piecewise_sampler(angle_histogram_data)
        overlap_sampler = create_piecewise_sampler(overlap_histogram_data)
        
        # 对每一列分别应用各自的采样器
        impact_velocity = velocity_sampler(uniform_samples[:, 0])
        impact_angle = angle_sampler(uniform_samples[:, 1])
        overlap = overlap_sampler(uniform_samples[:, 2])
        
    # 如果某个采样恰好取到0或者±100%附近的值 直接设为100%
    overlap = np.where((np.abs(overlap) > 0.99) | (np.abs(overlap) < 0.02), 1.0, overlap)

    # 对于重叠率绝对值在0.25~0.3之间的样本，强制碰撞角度与重叠率异号且绝对值>30°
    mask = (np.abs(overlap) >= 0.25) & (np.abs(overlap) < 0.3)
    for i in np.where(mask)[0]:
        # 如果角度与重叠率同号或角度绝对值<=30，则重新采样角度
        while np.sign(impact_angle[i]) == np.sign(overlap[i]) or np.abs(impact_angle[i]) <= 30:
            # 重新采样角度（均匀分布在[-45, 45]，异号且绝对值>30）
            if overlap[i] > 0:
                if overlap[i] <= 0.26:
                    impact_angle[i] = np.random.uniform(-45, -40)
                elif overlap[i] <= 0.28:
                    impact_angle[i] = np.random.uniform(-45, -35)
                else:
                    impact_angle[i] = np.random.uniform(-45, -30)
            else:
                if overlap[i] >= -0.26:
                    impact_angle[i] = np.random.uniform(40, 45)
                elif overlap[i] >= -0.28:
                    impact_angle[i] = np.random.uniform(35, 45)
                else:
                    impact_angle[i] = np.random.uniform(30, 45)

    # 创建DataFrame
    data = {
        'impact_velocity': impact_velocity,
        'impact_angle': impact_angle,
        'overlap': overlap,
    }
    
    # 添加nan值占位约束系统参数
    param_names = [
        'occupant_type', 'll1', 'll2', 'btf', 'pp', 'plp',
        'lla_status', 'llattf', 'dz', 'ptf', 'aft', 'aav_status',
        'ttf', 'sp', 'recline_angle'
    ]
    
    for param in param_names:
        data[param] = np.full(n_samples, np.nan)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 添加额外的列
    if case_ids is None:
        case_ids = np.arange(1, n_samples + 1)
    
    df.insert(0, 'case_id', case_ids)
    df.insert(1, 'have_run', False) # 后面再填充True/False
    df.insert(2, 'is_pulse_ok', np.full(n_samples, np.nan)) # 后面再填充True/False
    df.insert(3, 'is_injury_ok', np.full(n_samples, np.nan)) # 后面再填充True/False
    
    # 保存结果
    if filename.endswith('npz'):
        np.savez_compressed(filename, **{col: df[col].values for col in df.columns})
    elif filename.endswith('csv'):
        df.to_csv(filename, index=False)
    else:
        raise ValueError("Unsupported file format. Use '.npz' or '.csv'.")
    
    print(f"碰撞工况参数采样完成，结果已保存至 '{filename}'")
    print(f"总样本数: {n_samples}")
    
    # 可视化前几个样本
    print("\n前5个样本:")
    print(df.head())
    
    return filename

sample_collision_params(n_samples=3000, skip_points=3000, method='uniform', filename=r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_test1.csv', seed=20252025, case_ids=np.arange(1, 3001))

# %% 第二部分：对剩余约束系统参数（包括座椅、乘员体征等参数）进行均匀采样
import numpy as np
import pandas as pd
from scipy.stats import qmc
import random
from scipy.interpolate import RectBivariateSpline

class BTFSampler:
    """
    根据速度和重叠率，在一个动态范围内随机采样BTF值。

    该采样器基于一个离散的BTF推荐值表，通过二维插值和三角分布随机化
    来为任意输入工况生成一个合理的BTF值。

    新版逻辑：即使输入工况为表格中的精确点，也会根据其邻居节点确定一个
    随机范围，从而保证所有采样都具有随机性。
    """
    def __init__(self):
        # 1. 数据准备
        self.speeds = np.array([25, 35, 45, 55, 65])
        self.overlaps = np.array([25, 50, 75, 100])
        self.btf_values = np.array([
            [100, 45, 35, 30],
            [50,  35, 30, 25],
            [45,  25, 25, 20],
            [40,  15, 20, 15],
            [35,  10, 15, 10]
        ])
        self.interpolator = RectBivariateSpline(self.speeds, self.overlaps, self.btf_values, kx=1, ky=1)

    def _get_bounding_box_values(self, speed, overlap):
        """找到输入点所在单元格的四个角点的值。"""
        s_idx_high = np.searchsorted(self.speeds, speed)
        o_idx_high = np.searchsorted(self.overlaps, overlap)
        s_idx_low = max(0, s_idx_high - 1)
        o_idx_low = max(0, o_idx_high - 1)
        s_idx_high = min(len(self.speeds) - 1, s_idx_high)
        o_idx_high = min(len(self.overlaps) - 1, o_idx_high)
        
        return [
            self.btf_values[s_idx_low, o_idx_low],
            self.btf_values[s_idx_low, o_idx_high],
            self.btf_values[s_idx_high, o_idx_low],
            self.btf_values[s_idx_high, o_idx_high]
        ]

    def _get_neighbor_values(self, s_idx, o_idx):
        """获取一个网格点及其直接邻居（上/下/左/右）的值。"""
        num_speeds, num_overlaps = self.btf_values.shape
        neighbor_vals = [self.btf_values[s_idx, o_idx]] # Start with the point itself

        # Up neighbor
        if s_idx > 0:
            neighbor_vals.append(self.btf_values[s_idx - 1, o_idx])
        # Down neighbor
        if s_idx < num_speeds - 1:
            neighbor_vals.append(self.btf_values[s_idx + 1, o_idx])
        # Left neighbor
        if o_idx > 0:
            neighbor_vals.append(self.btf_values[s_idx, o_idx - 1])
        # Right neighbor
        if o_idx < num_overlaps - 1:
            neighbor_vals.append(self.btf_values[s_idx, o_idx + 1])
            
        return neighbor_vals

    def sample(self, speed: float, overlap_rate: float) -> float:
        """主采样函数"""
        abs_overlap = abs(overlap_rate)
        clamped_speed = np.clip(speed, self.speeds[0], self.speeds[-1])
        clamped_overlap = np.clip(abs_overlap, self.overlaps[0], self.overlaps[-1])

        # 检查输入是否为表格上的精确网格点
        is_on_grid_point = clamped_speed in self.speeds and clamped_overlap in self.overlaps

        if is_on_grid_point:
            # --- 新逻辑：处理精确表格中网格点 ---
            s_idx = np.where(self.speeds == clamped_speed)[0][0]
            o_idx = np.where(self.overlaps == clamped_overlap)[0][0]
            
            # 使用邻居节点的值来定义范围
            range_values = self._get_neighbor_values(s_idx, o_idx)
            min_btf = min(range_values)
            max_btf = max(range_values)
            
            # 最可能的值是该点本身的值
            center_btf = self.btf_values[s_idx, o_idx]
        else:
            # --- 原逻辑：处理网格之间的点 ---
            center_btf = self.interpolator(clamped_speed, clamped_overlap)[0, 0]
            bounding_values = self._get_bounding_box_values(clamped_speed, clamped_overlap)
            min_btf = min(bounding_values)
            max_btf = max(bounding_values)

        # --- 统一的随机采样步骤 ---
        if min_btf >= max_btf: # 使用>=以处理浮点数精度问题
            sampled_btf = min_btf
        else:
            sampled_btf = random.triangular(low=min_btf, high=max_btf, mode=center_btf)

        final_btf = np.clip(sampled_btf, 10, 100)
        return final_btf

def sample_restraint_params(filename, new_filename,  case_ids, n_samples=None, skip_points=16384, seed=20252025):
    """
    对约束系统参数进行采样，并填充到指定的文件中
    
    参数:
    - filename: 第一部分生成的文件名
    - case_ids: 需要填充的case_id列表
    - n_samples: 采样数量，默认为case_ids的长度
    - skip_points: 跳过的初始点数量
    - seed: 随机种子
    
    返回:
    - 新文件名
    """
    print(f"开始对约束系统参数进行采样...")
    
    # 确定采样数量
    if n_samples is None:
        n_samples = len(case_ids)
    
    print(f"  - 将对{n_samples}个case_id进行约束系统参数填充")
    print(f"  - 跳过初始点: {skip_points}")
    print(f"  - 随机种子: {seed}")
    
    # 加载之前的数据
    if filename.endswith('.npz'):
        with np.load(filename) as data:
            existing_data = {key: data[key] for key in data.files}
            is_npz = True
    elif filename.endswith('.csv'):
        existing_data = pd.read_csv(filename)
        is_npz = False
    else:
        raise ValueError("Unsupported file name format. Use '.npz' or '.csv'.")
    
    # 约束系统等参数
    param_dims = {
        'occupant_type':    0,
        'll1':              1,
        'll2':              2,
        'btf':              3, # BTF单独采样处理
        'pp':               4,
        'plp':              5,
        'lla_status':       6,
        'llattf_offset':    7,
        'dz':               8,
        'aft':              9,
        'aav_status':       10,
        'ttf_offset':       11,
        'sp':               12,
        'recline_angle':    13
    }
    # 初始化Sobol序列生成器
    sampler = qmc.Sobol(d=len(param_dims), scramble=True, seed=seed)
    btf_sampler = BTFSampler()

    # 跳过前面的初始点
    sampler.fast_forward(skip_points)
    
    # 生成 [0, 1) 范围内的标准样本点
    samples_unit_cube = sampler.random(n=n_samples)
    
    # 创建一个字典来存储最终的参数值
    results = {}
    
    # 获取碰撞工况参数，用于BTF的采样约束
    if is_npz:
        impact_velocity = {case_id: existing_data['impact_velocity'][i] for i, case_id in enumerate(existing_data['case_id'])}
        overlap = {case_id: existing_data['overlap'][i] for i, case_id in enumerate(existing_data['case_id'])}
    else:
        impact_velocity = dict(zip(existing_data['case_id'], existing_data['impact_velocity']))
        overlap = dict(zip(existing_data['case_id'], existing_data['overlap']))

    # 遍历每个样本点进行缩放
    for i in range(n_samples):
        sample = samples_unit_cube[i]
        case_id = case_ids[i]
        
        # --- 乘员体征参数 ---
        # 映射到 [1, 2, 3]
        occupant_type = np.floor(sample[param_dims['occupant_type']] * 3) + 1
        results.setdefault('occupant_type', []).append(int(occupant_type))
        
        # --- 安全带系统 ---
        # 安全带二级限力值ll2需要小于一级限力值ll1
        while True:
            # 拒绝采样方法
            ll1_candidate = sampler.random(1)[0, param_dims['ll1']] * (7.0 - 2.0) + 2.0 # ll1 [2, 7]kN
            ll2_candidate = sampler.random(1)[0, param_dims['ll2']] * (4.5 - 1.5) + 1.5 # ll2 [1.5, 4.5]kN

            if ll1_candidate > ll2_candidate:
                ll1_val = ll1_candidate
                ll2_val = ll2_candidate
                break
        
        results.setdefault('ll1', []).append(ll1_val)
        results.setdefault('ll2', []).append(ll2_val)

        # 预紧器点火时刻与碰撞速度和重叠率相关; 如果velocity或overlap缺失则btf仍然使用sobel采样，否则单独采样
        velocity = impact_velocity.get(case_id, None)
        overlap_rate = overlap.get(case_id, None)
        if velocity is None or overlap_rate is None or np.isnan(velocity) or np.isnan(overlap_rate):
            # 使用Sobol采样
            btf_val = sample[param_dims['btf']] * (100 - 10) + 10 # btf [10, 100]ms
        else:
            btf_val = btf_sampler.sample(velocity, overlap_rate)  # 采样BTF值
        results.setdefault('btf', []).append(btf_val)

        results.setdefault('pp', []).append(sample[param_dims['pp']] * (100 - 40) + 40) # pp [40, 100]mm
        # 计算PTF (确定性)
        results.setdefault('ptf', []).append(btf_val + 7.0) # ptf = btf + 7ms
        results.setdefault('plp', []).append(sample[param_dims['plp']] * (80 - 20) + 20) # plp [20, 80]mm
        results.setdefault('lla_status', []).append(int(np.floor(sample[param_dims['lla_status']] * 2))) # lla_status 0或1 0代表不切换二级限力 1代表切换二级限力(此时LLATTF生效)
        # 计算LLATTF
        llattf_offset_val = sample[param_dims['llattf_offset']] * 100 # llattf_offset [0, 100]ms
        results.setdefault('llattf', []).append(btf_val + llattf_offset_val) # llattf = btf + llattf_offset
        results.setdefault('dz', []).append(int(np.floor(sample[param_dims['dz']] * 4) + 1)) # dz 1, 2, 3, 4

        # --- 气囊系统 ---
        # 气囊点火时刻aft需要满足aft < 25 + btf
        while True:
            # 拒绝采样方法
            aft_candidate = sampler.random(1)[0, param_dims['aft']] * (100 - 10) + 10 # aft [10, 100]ms
            if aft_candidate < (25 + btf_val):
                aft_val = aft_candidate
                break
        results.setdefault('aft', []).append(aft_val)
        
        results.setdefault('aav_status', []).append(int(np.floor(sample[param_dims['aav_status']] * 2))) # aav_status 0或1 0代表不开启二级泄气孔 1代表开启二级泄气孔(此时TTF生效)

        # 计算TTF，需满足TTF > 0.5*BTF
        while True:
            ttf_offset_candidate = sampler.random(1)[0, param_dims['ttf_offset']] * 100 # ttf_offset [0, 100]ms
            ttf_val = aft_val + ttf_offset_candidate
            if ttf_val > 0.5 * btf_val:
                ttf_offset_val = ttf_offset_candidate
                break
        
        results.setdefault('ttf', []).append(aft_val + ttf_offset_val) # ttf = aft + ttf_offset
        
        # --- 座椅参数 ---
        # 根据乘员体型决定座椅位置范围
        sp_sample = sample[param_dims['sp']]
        if occupant_type == 1: # 5% 假人
            sp_val = sp_sample * (110 - 10) + 10 # sp [10, 110]mm
        elif occupant_type == 2: # 50% 假人
            sp_val = sp_sample * (80 - (-80)) + (-80) # sp [-80, 80]mm
        else: # 95% 假人
            sp_val = sp_sample * (40 - (-110)) + (-110) # sp [-110, 40]mm
        results.setdefault('sp', []).append(sp_val)
        results.setdefault('recline_angle', []).append(sample[param_dims['recline_angle']] * (15 - (-10)) + (-10)) # recline_angle [-10, 15]°
    
    # 将列表转换为Numpy数组
    for key in results:
        results[key] = np.array(results[key])
    
    # 更新原始数据
    if is_npz:
        # 获取case_id在原始数据中的索引
        case_id_indices = {}
        for i, case_id in enumerate(existing_data['case_id']):
            case_id_indices[case_id] = i
        
        # 创建一个新的数据字典，用于保存更新后的数据
        updated_data = {k: existing_data[k].copy() for k in existing_data}
        
        # 更新约束系统参数
        for i, case_id in enumerate(case_ids):
            if case_id in case_id_indices:
                idx = case_id_indices[case_id]
                for param_name in results:
                    updated_data[param_name][idx] = results[param_name][i]
    else:
        # CSV方式处理
        df = existing_data
        
        # 对每个case_id进行更新
        for i, case_id in enumerate(case_ids):
            idx = df[df['case_id'] == case_id].index
            if len(idx) > 0:
                idx = idx[0]
                for param_name in results:
                    df.at[idx, param_name] = results[param_name][i]
        
        updated_data = df
    
    # 保存更新后的数据
    if new_filename.endswith('.npz'):
        np.savez_compressed(new_filename, **updated_data)
    elif new_filename.endswith('.csv'):
        updated_data.to_csv(new_filename, index=False)
    else:
        raise ValueError("Unsupported new file name format. Use '.npz' or '.csv'.")

    print(f"约束系统参数采样并填充完成，结果已保存至 '{new_filename}'")
    
    # 打印一个样本作为示例
    if is_npz:
        print("\n--- 采样结果示例 (第一个case_id) ---")
        for key in results:
            print(f"{key:<20}: {results[key][0]:.4f}")
    else:
        print("\n更新后的前5个样本:")
        print(updated_data.head())

    return new_filename

distribution_file = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0917.csv'
new_filename = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0919.csv'

# 读取distribution_file中，is_pulse_ok为TRUE、且'occupant_type'还没有值的的case_id列，转为list，这部分作为填充的case_ids;
if distribution_file.endswith('.csv'):
    df = pd.read_csv(distribution_file)
    case_ids_to_fill = df[df['is_pulse_ok'] == True & df['occupant_type'].isnull()]['case_id'].tolist()
if distribution_file.endswith('.npz'):
    with np.load(distribution_file) as data:
        df = pd.DataFrame({key: data[key] for key in data.files})
        case_ids_to_fill = df[df['is_pulse_ok'] == True & df['occupant_type'].isnull()]['case_id'].tolist()

print(f"需要填充约束系统参数的case_id数量: {len(case_ids_to_fill)}")
print(f"部分case_id示例（开头和结尾）: {case_ids_to_fill[:10]} ... {case_ids_to_fill[-10:]}")

sample_restraint_params(filename=distribution_file, new_filename=new_filename, case_ids=case_ids_to_fill, skip_points=18000, seed=20252025)