# -*- coding: utf-8 -*-
# 仅判断是否存在nan时，仿真时长和结果步长影响文件大小
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
import os
import warnings
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, detrend
from scipy.signal.windows import tukey
# from numba import jit
from scipy import signal
warnings.filterwarnings('ignore')

g=9.81    #单位m/s2
xls_route = r'I:\000 LX\dataset0715\02\results_0918'
output_path = r'I:\000 LX\dataset0715\03\acc_data_0918'
params_path = r'I:\000 LX\dataset0715\02\distribution_0919.csv' 

if params_path.endswith('.npz'):
    params_npz = np.load(params_path, allow_pickle=True)
    params_df = pd.DataFrame({
        key: params_npz[key]
        for key in params_npz.files
    }).set_index('case_id')
elif params_path.endswith('.csv'):
    params_df = pd.read_csv(params_path)
    params_df.set_index('case_id', inplace=True)


# 如果输出文件夹不存在，则创建
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"创建输出文件夹: {output_path}")

# 寻找目标文件夹下所有xlsx文件
path_list = []
for file in os.listdir(xls_route):
    if file.startswith('~$'):
        continue
    elif file.endswith('.xlsx'):
        path_list.append(file)
print("="*50)
print(f"发现 {len(path_list)} 个xlsx文件")
print("="*50)

# case_xxx.xlsx， 先提取中间的数字，作为case的编号同时排序path_list
path_list.sort(key=lambda x: int(x.split('.')[0].split('_')[1]))

def CFC_filter(time_filter, signal, CFC=60):
    """
    根据 ISO 6487:2015 规范，使用 Scipy 库对信号进行CFC无相位滤波。
    此实现利用 butter 和 filtfilt 函数, 实现了一个零相位（无延迟）、四阶巴特沃斯低通滤波器。
    此算法明确规定适用于 CFC 60 和 CFC 180。

    参数:
    ----------
    time_filter : array_like
        时间序列数组，单位为秒(s)。必须是等间隔采样。
    signal : array_like
        需要滤波的信号数组，与 time_filter 对应。
    CFC : int, 可选
        通道频率等级 (Channel Frequency Class)。仅支持 60 或 180。默认为 60。

    返回:
    -------
    filtered_signal : numpy.ndarray
        滤波后的信号数组。
    """
    # --- 1. 参数计算与验证 ---
    time_filter = np.asarray(time_filter, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    if time_filter.shape != signal.shape:
        raise ValueError("时间和信号数组的长度必须一致。")

    sample_rate = 1.0 / np.mean(np.diff(time_filter))

    if CFC not in [60, 180]:
        raise ValueError("根据 ISO 6487 附录A，此算法仅适用于 CFC 60 和 CFC 180。")

    # --- 2. 设计巴特沃斯滤波器 ---
    if CFC == 60:
        f_cutoff = 100.0  # Hz
    else:  # CFC == 180
        f_cutoff = 300.0  # Hz

    # 根据 ISO 6487 附录A，这是一个 "four-pole phaseless digital filter"。
    # filtfilt 通过前向-后向滤波实现，其幅值响应等效于阶数加倍。
    # 因此，一个二阶(N=2)巴特沃斯滤波器，经 filtfilt 处理后，其效果等效于一个四阶(four-pole)滤波器的幅值响应。
    b, a = butter(N=2, Wn=f_cutoff, btype='lowpass', analog=False, fs=sample_rate)

    # --- 3. 应用无相位滤波器 ---
    # filtfilt 实现了标准要求的“无相位”特性，并内置了高效的边界处理。
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal

# CSV保存函数
def save_csv(data_dict, filename, output_dir=None):
    """优化的CSV保存函数，避免重复DataFrame创建"""
    if output_dir:
        filename = os.path.join(output_dir, filename)
    df = pd.DataFrame(data_dict)
    df.to_csv(filename, sep='\t', index=False, header=False,
              quoting=csv.QUOTE_NONE, quotechar="", escapechar="\t")

bad_case_ids_list = []

for file in path_list:
    case_id = int(file.split('.')[0].split('_')[1])  # 提取case编号
    # if case_id <= 11:  # 跳过已经处理过的case
    #     continue
    print(f'Processing case {case_id}...')
    # ------------------------------------------------------------
    impact_velocity = params_df.loc[case_id, 'impact_velocity']  # km/h
    impact_angle = params_df.loc[case_id, 'impact_angle']  # deg
    overlap = params_df.loc[case_id, 'overlap']  # -1~1
    if abs(overlap) < 0.25:
        print(f"**Warning: Overlap {overlap} out of range in case {case_id}. Skipping this case.")
        continue
    if 0.25 <= abs(overlap) < 0.3:
        # 如果碰撞角度与之同号，跳过该case
        if impact_angle * overlap > 0:
            print(f"**Warning: Overlap {overlap} and Impact Angle {impact_angle} have the same sign in case {case_id}. Skipping this case.")
            continue
        # 如果碰撞角度绝对值<=30，跳过该case
        if abs(impact_angle) <= 30:
            print(f"**Warning: Overlap {overlap} , while Impact Angle {impact_angle} out of range in case {case_id}. Skipping this case.")
            continue
    # ------------------------------------------------------------

    # 先获取文件大小，一般小于9900KB的xlsx文件会含有nan值，加入bad_case_ids_list
    file_size_kb = os.path.getsize(os.path.join(xls_route, file)) / 1024  # 获取文件大小, 单位KB
    #if file_size_kb < 9900: # 200ms 0.01ms则为9900KB
    if file_size_kb < 14000: # 150ms 0.005ms则为14000KB
        print(f"**Warning: {file_size_kb} KB of case {case_id}. Skipping the case.")
        bad_case_ids_list.append(case_id)
        continue

    # 只读取需要的列，减少内存占用和读取时间
    usecols = [0, 1, 2, 3, 17, 19, 37]  # 只读取需要的列：时间、xyz位置、vx、vy、omg_z
    data = pd.read_excel(os.path.join(xls_route, file), engine='openpyxl', usecols=usecols)
    data_values = data.values  # 转换为numpy数组提高访问速度
    
    # 预分配所有需要的变量，避免重复索引
    tim = data_values[:, 0] * 0.001  # 将ms换算成s
    t_ms = data_values[:, 0]  # ms，用于后续计算
    x, y, z = data_values[:, 1], data_values[:, 2], data_values[:, 3]  # mm
    vx, vy = data_values[:, 4], data_values[:, 5]  # m/s
    omg_z = data_values[:, 6]  # rad/s
  

    # 如果data_values存在nan值，直接跳过并打印提示. 异常case主要是含有nan，但文件长度倒是固定
    if np.isnan(data_values).any():
        print(f"**Warning: NaN values found in case {case_id}. Skipping this case.")
        # delta_v[case_id-1] = 0  # 设置该case的delta_v为0
        bad_case_ids_list.append(case_id)
        continue
    
    # ****** X方向 ******
    vx_filtered = CFC_filter(tim, vx, 60) # m/s
    ax = np.gradient(vx_filtered, tim)  # m/s^2
    ax_filtered = -CFC_filter(tim, ax, 60) # 注意取负值方向

    if np.abs(ax_filtered).max() < 10:  # 如果最大加速度小于10m/s2，说明数据有问题，跳过
        print(f"**Warning: Max |ax| < 10 m/s² in case {case_id}. Skipping this case.")
        bad_case_ids_list.append(case_id)
        continue

    x_name='x'+str(case_id)+'.csv' 
    save_csv({'XI': tim, 'YI': ax_filtered}, x_name, output_path)

    # ****** Y方向 ******
    vy_filtered = CFC_filter(tim, vy, 60)
    ay = np.gradient(vy_filtered, tim)  # m/s^2
    ay_filtered = CFC_filter(tim, ay, 60) # 
    y_name='y'+str(case_id)+'.csv'
    save_csv({'XI': tim, 'YI': ay_filtered}, y_name, output_path)

    #  # ****** Z方向 角加速度 ******
    #对应元素点乘np.multiply，对应元素矩阵乘np.dot
    #pos=[-2338,775,341]  #算VCS中的半径变化
    pos = [1638.13, -799.49, -18.65]
    xt = x + pos[0]
    yt = y + pos[1]
    zt = z + pos[2]   #负值是因为重力沉降
    rt = np.sqrt(xt*xt + yt*yt + zt*zt)   #mm 修复矩阵运算错误
    const = np.sqrt(2475**2 + 775**2 + 341**2)   #算MADYMO中的半径值 mm。因为两个原点在x轴上有一定偏移
    temp = omg_z * rt * 0.001   #m/s 使用向量化操作

    temp_filtered = CFC_filter(tim, temp, 60)

    omg_new = temp_filtered / (const * 0.001)   #rad/s
    rot_acc = np.gradient(omg_new, tim)   #rad/s2
    az = CFC_filter(tim, rot_acc, 60)

    z_name = 'z' + str(case_id) + '.csv' 
    save_csv({'XI': tim, 'YI': az}, z_name, output_path)

    # delta_v[case_id-1] = np.mean(vx[0:10]) - np.mean(vx[-11:-1])   #m/s
    # np.save("delta_vx.npy", delta_v)
    # print(f'case_{case_id}.xlsx calculate successfully, with delta_vx: {delta_v[case_id-1]}') 

# 将bad_case_ids_list保存到npy文件
# np.save("bad_case_ids_addition600.npy", np.array(bad_case_ids_list))


