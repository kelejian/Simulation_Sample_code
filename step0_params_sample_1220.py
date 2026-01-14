# %% 第一部分：对三个碰撞工况参数进行采样，用于VCS碰撞波形仿真
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

        # 重叠率 ('overlap') 数据, 调整为 (-1, -0.25]∪[0.25, 1] 范围
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
    # 创建独立的随机数生成器用于角度拒绝采样
    angle_rejection_rng = np.random.Generator(np.random.PCG64(seed + 999))
    
    mask = (np.abs(overlap) >= 0.25) & (np.abs(overlap) < 0.3)
    for i in np.where(mask)[0]:
        # 如果角度与重叠率同号或角度绝对值<=30，则重新采样角度
        while np.sign(impact_angle[i]) == np.sign(overlap[i]) or np.abs(impact_angle[i]) <= 30:
            # 重新采样角度（均匀分布在[-45, 45]，异号且绝对值>30）
            if overlap[i] > 0:
                if overlap[i] <= 0.26:
                    impact_angle[i] = angle_rejection_rng.uniform(-45, -40)
                elif overlap[i] <= 0.28:
                    impact_angle[i] = angle_rejection_rng.uniform(-45, -35)
                else:
                    impact_angle[i] = angle_rejection_rng.uniform(-45, -30)
            else:
                if overlap[i] >= -0.26:
                    impact_angle[i] = angle_rejection_rng.uniform(40, 45)
                elif overlap[i] >= -0.28:
                    impact_angle[i] = angle_rejection_rng.uniform(35, 45)
                else:
                    impact_angle[i] = angle_rejection_rng.uniform(30, 45)

    # 创建DataFrame
    data = {
        'impact_velocity': impact_velocity,
        'impact_angle': impact_angle,
        'overlap': overlap,
    }
    
    # 添加nan值占位约束系统参数和损伤值标签
    param_names = [
        'occupant_type', 'll1', 'll2', 'btf', 'pp', 'plp',
        'lla_status', 'llattf', 'dz', 'ptf', 'aft', 'aav_status',
        'ttf', 'sp', 'recline_angle',
        'HIC15', 'Dmax', 'Nij'
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

sample_collision_params(n_samples=7000, skip_points=5048, method='non_uniform', filename=r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_test1.csv', seed=20250923, case_ids=np.arange(3001, 3001+7000))

# %% 第二部分（20251220版代码）：对约束系统参数进行采样，用于MADYMO乘员损伤仿真
import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.interpolate import RectBivariateSpline
class BTFSampler:
    """
    根据速度和重叠率，在一个动态范围内随机采样BTF值。

    该采样器基于一个离散的BTF推荐值表，通过二维插值和三角分布随机化
    来为任意输入工况生成一个合理的BTF值。即使输入工况为表格中的精确点，
    也会根据其邻居节点确定一个随机范围，从而保证所有采样都具有随机性。
    """
    def __init__(self, seed=None):
        # 数据准备：BTF推荐值表
        # 行：速度 [25, 35, 45, 55, 65] kph
        # 列：重叠率绝对值 [25%, 50%, 75%, 100%]
        self.speeds = np.array([25, 35, 45, 55, 65])
        self.overlaps = np.array([25, 50, 75, 100])
        self.btf_values = np.array([
            [100, 45, 35, 30],  # 25 kph (OFF时取100)
            [50,  35, 30, 25],  # 35 kph
            [45,  25, 25, 20],  # 45 kph
            [40,  15, 20, 15],  # 55 kph
            [35,  10, 15, 10]   # 65 kph
        ])
        self.interpolator = RectBivariateSpline(self.speeds, self.overlaps, self.btf_values, kx=1, ky=1)
        
        # 创建独立的随机数生成器
        self.rng = np.random.Generator(np.random.PCG64(seed if seed is not None else 12345))

    def _get_bounding_box_values(self, speed, overlap):
        """找到输入点所在单元格的四个角点的值."""
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
        """获取一个网格点及其直接邻居（上/下/左/右）的值."""
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
        # 将重叠率（-1到1）转换为绝对值的百分比（0到100）
        abs_overlap_percent = abs(overlap_rate) * 100.0
        
        clamped_speed = np.clip(speed, self.speeds[0], self.speeds[-1])
        clamped_overlap = np.clip(abs_overlap_percent, self.overlaps[0], self.overlaps[-1])

        # 检查输入是否为表格上的精确网格点
        is_on_grid_point = clamped_speed in self.speeds and clamped_overlap in self.overlaps

        if is_on_grid_point:
            # --- 处理精确表格中网格点 ---
            s_idx = np.where(self.speeds == clamped_speed)[0][0]
            o_idx = np.where(self.overlaps == clamped_overlap)[0][0]
            
            # 使用邻居节点的值来定义范围
            range_values = self._get_neighbor_values(s_idx, o_idx)
            min_btf = min(range_values)
            max_btf = max(range_values)
            
            # 最可能的值是该点本身的值
            center_btf = self.btf_values[s_idx, o_idx]
        else:
            # --- 处理网格之间的点 ---
            center_btf = self.interpolator(clamped_speed, clamped_overlap)[0, 0]
            bounding_values = self._get_bounding_box_values(clamped_speed, clamped_overlap)
            min_btf = min(bounding_values)
            max_btf = max(bounding_values)

        # --- 统一的随机采样步骤 ---
        if min_btf >= max_btf: # 使用>=以处理浮点数精度问题
            sampled_btf = min_btf
        else:
            # 使用独立的随机数生成器进行三角分布采样
            sampled_btf = self.rng.triangular(left=min_btf, mode=center_btf, right=max_btf)

        final_btf = np.clip(sampled_btf, 10, 100)
        return final_btf


def create_piecewise_sampler(histogram_data):
    """
    创建分段非均匀采样器，用于加权采样。
    
    参数:
    - histogram_data: 列表，每个元素为 [区间下限, 区间上限, 相对密度]
      - 相对密度表示该区间的采样密度，值越大该区间被采样的概率越高
      - 实际概率 = (区间宽度 × 相对密度) / (所有区间面积之和)
      - 例如：[2.0, 3.0, 80.0] 表示在 [2.0, 3.0) 区间的相对密度为 80.0
    
    返回:
    - sampler函数：输入[0,1)均匀样本，输出对应分布的采样值
    """
    bins = np.array(histogram_data)
    x_mins, x_maxs, densities = bins[:, 0], bins[:, 1], bins[:, 2]
    
    widths = x_maxs - x_mins
    areas = widths * densities  # 每个区间的面积 = 宽度 × 密度
    bin_probabilities = areas / np.sum(areas)  # 归一化得到概率
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


def sample_restraint_params(filename, new_filename, case_ids, n_samples=None, skip_points=2048, seed=20252025):
    """
    对约束系统参数进行采样，并填充到指定的文件中，区分主副驾侧）
    
    参数:
    - filename: 输入的distribution文件名（.csv或.npz）
    - new_filename: 输出的新distribution文件名
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
    
    print(f"  - 将对 {n_samples} 个case_id进行约束系统参数填充")
    print(f"  - 跳过初始点: {skip_points}")
    print(f"  - 随机种子: {seed}")
    
    # ==================== 加载数据 ====================
    if filename.endswith('.npz'):
        with np.load(filename) as data:
            existing_data = pd.DataFrame({key: data[key] for key in data.files})
        is_npz = True
    elif filename.endswith('.csv'):
        existing_data = pd.read_csv(filename)
        is_npz = False
    else:
        raise ValueError("Unsupported file format. Use '.npz' or '.csv'.")
    
    # 设置case_id为索引以便快速查找
    existing_data.set_index('case_id', drop=False, inplace=True)
    
    # ==================== 定义参数采样维度 ====================
    param_dims = {
        'OT':           0,   # 乘员体型 [1, 2, 3]
        'LL1':          1,   # 一级限力值 [2.0, 7.0] kN，非均匀采样
        'LL2':          2,   # 二级限力值 [1.5, 4.5] kN，非均匀采样
        'BTF':          3,   # 预紧器点火时刻（用于备用采样）
        'LLATTF_offset': 4,  # 二级限力切换时间偏移 [0, 100] ms
        'AFT':          5,   # 气囊点火时刻 [10, 100] ms
        'SP':           6,   # 座椅前后位置（归一化采样，后续根据体型和主副驾映射）
        'RA':           7,   # 座椅靠背角度（归一化采样，后续离散化）
    }
    
    # ==================== 初始化采样器 ====================
    # Sobol序列采样器
    sobol_sampler = qmc.Sobol(d=len(param_dims), scramble=True, seed=seed)
    sobol_sampler.fast_forward(skip_points)
    samples_unit_cube = sobol_sampler.random(n=n_samples)
    
    # BTF采样器（基于速度和重叠率的插值采样）
    btf_sampler = BTFSampler(seed=seed + 888)
    
    # 拒绝采样用的独立随机数生成器
    rejection_rng = np.random.Generator(np.random.PCG64(seed + 666))
    
    # LL1非均匀采样器：[3.0, 5.0] kN 区间加权到75%~80%
    ll1_histogram_data = [
        [2.0, 3.0, 8.0], 
        [3.0, 4.0, 35.0],
        [4.0, 5.0, 40.0], 
        [5.0, 6.0, 10.0],
        [6.0, 7.0, 7.0],
    ]
    ll1_sampler = create_piecewise_sampler(ll1_histogram_data)
    
    # LL2非均匀采样器：[1.5, 3.0] kN 区间加权到75~80%
    ll2_histogram_data = [
        [1.5, 2.0, 30.0], 
        [2.0, 3.0, 40.0], 
        [3.0, 4.0, 17.0], 
        [4.0, 4.5, 13.0],
    ]
    ll2_sampler = create_piecewise_sampler(ll2_histogram_data)
    
    # OT非均匀采样器：OT=1(30%), OT=2(40%), OT=3(30%)
    # 使用离散值采样，每个OT值占据单位宽度区间
    ot_histogram_data = [
        [0.0, 1.0, 30.0],   # OT=1 (5th假人)，概率=30%
        [1.0, 2.0, 40.0],   # OT=2 (50th假人)，概率=40%
        [2.0, 3.0, 30.0],   # OT=3 (95th假人)，概率=30%
    ]
    ot_sampler = create_piecewise_sampler(ot_histogram_data)
    
    # ==================== 准备碰撞工况参数映射 ====================
    impact_velocity_map = dict(zip(existing_data['case_id'], existing_data['impact_velocity']))
    overlap_map = dict(zip(existing_data['case_id'], existing_data['overlap']))
    is_driver_side_map = dict(zip(existing_data['case_id'], existing_data['is_driver_side']))
    
    # ==================== 定义SP和RA的范围映射 ====================
    # SP范围：根据主副驾侧和乘员体型确定
    # 主驾 (is_driver_side=1): 5th: [+20, +110], 50th: [-80, +80], 95th: [-110, +20]
    # 副驾 (is_driver_side=0): 5th/50th: [-110, +110], 95th: [-110, +49]
    SP_RANGES = {
        # (is_driver_side, OT): (min_sp, max_sp)
        (1, 1): (20, 110),     # 主驾 5th
        (1, 2): (-40, 60),     # 主驾 50th 20260114调整
        (1, 3): (-110, 20),    # 主驾 95th
        (0, 1): (-110, 110),   # 副驾 5th
        (0, 2): (-110, 110),   # 副驾 50th
        (0, 3): (-110, 49),    # 副驾 95th
    }
    
    # RA离散值：主驾 [15, 20, 25, 30]°，副驾 [20, 25, 30, 35, 40]°
    RA_VALUES = {
        1: np.array([15, 20, 25, 30]),  # 主驾 20260114调整
        0: np.array([20, 25, 30, 35, 40]),  # 副驾
    }
    # ==================== 定义DZ与OT的映射 ====================
    # DZ与OT的映射关系
    DZ_MAP = {
        1: 1,  # 5th假人 -> DZ=1
        2: 3,  # 50th假人 -> DZ=3
        3: 4,  # 95th假人 -> DZ=4
    }
    
    # ==================== 开始采样 ====================
    results = {
        'OT': [], 'LL1': [], 'LL2': [], 'BTF': [], 'LLATTF': [],
        'DZ': [], 'AFT': [], 'SP': [], 'RA': [], 'PTF': []
    }
    
    for i in range(n_samples):
        sample = samples_unit_cube[i]
        case_id = case_ids[i]
        
        # 获取该case的碰撞工况参数和主副驾标识
        velocity = impact_velocity_map.get(case_id, np.nan)
        overlap_rate = overlap_map.get(case_id, np.nan)
        is_driver_side = is_driver_side_map.get(case_id, 1)  # 默认主驾
        
        # -------------------- 乘员体型 OT (非均匀采样) --------------------
        # OT=1(30%), OT=2(40%), OT=3(30%)
        ot_continuous = ot_sampler(sample[param_dims['OT']])  # 返回值在[0, 3)范围内
        ot_val = int(np.floor(ot_continuous)) + 1  # 转换为1, 2, 3
        ot_val = max(min(ot_val, 3), 1)  # 确保在[1, 3]范围内
        results['OT'].append(ot_val)
        
        # -------------------- D环高度 DZ (不采样，由OT确定) --------------------
        dz_val = DZ_MAP[ot_val]
        results['DZ'].append(dz_val)
        
        # -------------------- 安全带一级限力值 LL1 (非均匀采样) --------------------
        ll1_val = ll1_sampler(sample[param_dims['LL1']])
        
        # -------------------- 安全带二级限力值 LL2 (非均匀采样，需满足 LL2 < LL1) --------------------
        ll2_val = ll2_sampler(sample[param_dims['LL2']])
        
        # 拒绝采样确保 LL1 > LL2
        while ll1_val <= ll2_val:
            ll1_val = ll1_sampler(rejection_rng.random())
            ll2_val = ll2_sampler(rejection_rng.random())
        
        results['LL1'].append(float(ll1_val))
        results['LL2'].append(float(ll2_val))
        
        # -------------------- 预紧器点火时刻 BTF --------------------
        if np.isnan(velocity) or np.isnan(overlap_rate):
            # 碰撞工况参数缺失时使用Sobol采样
            btf_val = sample[param_dims['BTF']] * (100 - 10) + 10
        else:
            # 基于碰撞工况的插值采样
            btf_val = btf_sampler.sample(velocity, overlap_rate)
        results['BTF'].append(float(btf_val))
        
        # -------------------- 腰部预紧器点火时间 PTF (确定性计算) --------------------
        ptf_val = btf_val + 7.0
        results['PTF'].append(float(ptf_val))
        
        # -------------------- 二级限力切换时间 LLATTF --------------------
        # LLATTF = BTF + offset，其中 offset ∈ [0, 100] ms
        # 如果 LLATTF >= 150，则设为 150（代表不切换二级限力）
        # 额外规则：以5%概率直接设为150ms，确保不切换二级限力的样本占比大约5%~6%
        llattf_150ms_prob = 0.05
        if rejection_rng.random() < llattf_150ms_prob:
            # 直接设为150ms（不切换二级限力）
            llattf_val = 150.0
        else:
            # 正常采样：LLATTF = BTF + offset
            llattf_offset = sample[param_dims['LLATTF_offset']] * 100  # [0, 100] ms
            llattf_val = btf_val + llattf_offset
            if llattf_val >= 150:
                llattf_val = 150.0
        results['LLATTF'].append(float(llattf_val))
        
        # -------------------- 气囊点火时刻 AFT --------------------
        # 约束: AFT < BTF + 25
        aft_val = sample[param_dims['AFT']] * (100 - 10) + 10  # [10, 100] ms
        max_aft = btf_val + 25 - 0.001  # 留一点余量
        
        # 拒绝采样确保 AFT < BTF + 25
        while aft_val >= (btf_val + 25):
            aft_val = rejection_rng.uniform(10, min(100, max_aft))
        results['AFT'].append(float(aft_val))
        
        # -------------------- 座椅前后位置 SP (区分主副驾和体型) --------------------
        sp_range = SP_RANGES.get((int(is_driver_side), ot_val), (-110, 110))
        sp_min, sp_max = sp_range
        sp_val = sample[param_dims['SP']] * (sp_max - sp_min) + sp_min
        results['SP'].append(float(sp_val))
        
        # -------------------- 座椅靠背角度 RA (离散化采样，区分主副驾) --------------------
        ra_options = RA_VALUES.get(int(is_driver_side), RA_VALUES[1])
        # 将[0,1)均匀样本映射到离散档位
        ra_idx = int(np.floor(sample[param_dims['RA']] * len(ra_options)))
        ra_idx = max(min(ra_idx, len(ra_options) - 1), 0)  # 确保索引不越界,在0到len-1之间
        ra_val = ra_options[ra_idx]
        results['RA'].append(int(ra_val))
    
    # ==================== 转换为NumPy数组 ====================
    for key in results:
        results[key] = np.array(results[key])
    
    # ==================== 更新DataFrame ====================
    for i, case_id in enumerate(case_ids):
        if case_id in existing_data.index:
            for param_name in results:
                existing_data.at[case_id, param_name] = results[param_name][i]
    
    # 重置索引
    existing_data.reset_index(drop=True, inplace=True)
    
    # ==================== 保存结果 ====================
    if new_filename.endswith('.npz'):
        np.savez_compressed(new_filename, **{col: existing_data[col].values for col in existing_data.columns})
    elif new_filename.endswith('.csv'):
        existing_data.to_csv(new_filename, index=False)
    else:
        raise ValueError("Unsupported file format. Use '.npz' or '.csv'.")
    
    print(f"约束系统参数采样并填充完成，结果已保存至 '{new_filename}'")
    print(f"  - 采样参数: OT, LL1, LL2, BTF, LLATTF, DZ, AFT, SP, RA, PTF")
    return new_filename


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    distribution_file = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0113.csv'
    new_filename = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0114.csv'
    
    # 读取需要填充的case_id列表，可选只采样主驾侧的或副驾侧的
    # 条件：is_pulse_ok为True 且 OT为空（未采样过）
    driver_side_only = 1  # 设置为1表示只采样主驾侧，0表示只采样副驾侧，None表示采样所有
    if distribution_file.endswith('.csv'):
        df = pd.read_csv(distribution_file)
        if driver_side_only is not None:
            df = df[df['is_driver_side'] == driver_side_only]
        case_ids_to_fill = df[(df['is_pulse_ok'] == True) & (df['OT'].isnull())]['case_id'].tolist()
    elif distribution_file.endswith('.npz'):
        with np.load(distribution_file) as data:
            df = pd.DataFrame({key: data[key] for key in data.files})
            if driver_side_only is not None:
                df = df[df['is_driver_side'] == driver_side_only]
            case_ids_to_fill = df[(df['is_pulse_ok'] == True) & (df['OT'].isnull())]['case_id'].tolist()
    else:
        raise ValueError("Unsupported file format.")
    if driver_side_only == 1:
        print("仅对主驾侧进行约束系统参数采样。")
    elif driver_side_only == 0:
        print("仅对副驾侧进行约束系统参数采样。")
    else:
        print("对主副驾侧都进行约束系统参数采样。")
    print(f"需要填充约束系统参数的case_id数量: {len(case_ids_to_fill)}")
    
    if len(case_ids_to_fill) > 0:
        sample_restraint_params(
            filename=distribution_file,
            new_filename=new_filename,
            case_ids=case_ids_to_fill,
            skip_points=2048,
            seed=20251220
        )
    else:
        print("没有需要填充的case_id，跳过采样。")

# %%
# %%
