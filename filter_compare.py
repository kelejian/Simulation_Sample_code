import numpy as np
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
import csv
import math
import warnings
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, detrend
from scipy.signal.windows import tukey
warnings.filterwarnings('ignore')

g=9.81    #单位m/s2
# ------------------------------------------------------------------------------------
xls_route = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\new模型_全宽正碰结果'

output_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\new模型_全宽正碰结果'
# ------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------------
# 旧的CFC滤波器实现，有问题
def CFC_filter(time_filter,acceleration,CFC):
    # 确保输入数据类型为float64
    time_filter = np.asarray(time_filter, dtype=np.float64)
    acceleration = np.asarray(acceleration, dtype=np.float64)
    acce=np.zeros(acceleration.shape)
    T=np.max(time_filter)/len(time_filter)
    wd=2*math.pi*CFC*1.25/0.6
    wa=math.tan(wd*T/2)
    cons=2**0.5
    a0=wa**2/(1+cons*wa+wa**2)
    a1=2*a0
    a2=a0
    b1=2*(1-wa**2)/(1+cons*wa+wa**2)
    b2=(-1+cons*wa-wa**2)/(1+cons*wa+wa**2)
    acce[0]=acceleration[0]
    acce[1]=acceleration[1]
    for i in range(2,len(time_filter)):
        acce[i]=a0*acceleration[i]+a1*acceleration[i-1]+a2*acceleration[i-2]+b1*acce[i-1]+b2*acce[i-2]
    return acce

def LX_butter_lowpass_filter(data, cutoff, fs, order=2):  
    """应用低通滤波器"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def LX_filter(time_data, vel_data):

    def butter_lowpass_filter(data, cutoff, fs, order=2):  
        """应用低通滤波器"""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def process_single_group(time_data, vel_data):
        """处理单组数速度数滤波后求导，得到加速度再滤波一次"""
        # # 预处理速度数据
        # vel_data = -1 * vel_data
        
        # 计算采样频率
        avg_dt = np.mean(np.diff(time_data))    # 单位 ms
        fs = 1000 / avg_dt if avg_dt > 0 else 1000  # 采样频率
        
        # 滤波处理
        cutoff = 300  # 截止频率
        vel_filtered = butter_lowpass_filter(vel_data, cutoff, fs)
        # vel_filtered = butter_lowpass_filter(vel_data, 100, fs)
        
        # 计算加速度
        acc_data = np.zeros_like(vel_filtered)
        if len(time_data) > 2:
            for i in range(1, len(time_data) - 1):
                dt = (time_data[i+1] - time_data[i-1]) / 2000
                if dt != 0:
                    acc_data[i] = (vel_filtered[i+1] - vel_filtered[i-1]) / (2 * dt)
            
            # 处理端点
            if len(time_data) > 1:
                dt_first = (time_data[1] - time_data[0]) / 1000
                if dt_first != 0:
                    acc_data[0] = (vel_filtered[1] - vel_filtered[0]) / dt_first
                
                dt_last = (time_data[-1] - time_data[-2]) / 1000
                if dt_last != 0:
                    acc_data[-1] = (vel_filtered[-1] - vel_filtered[-2]) / dt_last
        
        # 加速度滤波
        acc_data_filtered = butter_lowpass_filter(acc_data, cutoff/2, fs) # 单位 m/s^2
        # acc_data_filtered = butter_lowpass_filter(acc_data, 100, fs) # 单位 m/s^2
        # acc_data_filtered = -1 * acc_data_filtered
        # acc_data_filtered_g = acc_data_filtered / 9.81
        return acc_data_filtered

    acc_data_filtered = process_single_group(time_data, vel_data)
    return acc_data_filtered

def CFC_filter_ISO_butterworth(time_filter, signal, CFC=60): # 最好用，最平滑
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
    # print(sample_rate)

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

def CFC_filter_SAE(time_filter, signal, CFC=60, aggressiveness=0.25): # 有边界伪影
    """
    根据 SAE J211-1 附录C 的规范，使用频域FFT方法对信号进行CFC无相位滤波。
    此版本增加了 'aggressiveness' 参数，用于控制滤波强度。

    参数:
    ----------
    time_filter : array_like
        时间序列数组，单位为秒(s)。必须是等间隔采样。
    signal : array_like
        需要滤波的信号数组，与 time_filter 对应。
    CFC : int, 可选
        通道频率等级 (Channel Frequency Class)。支持 60, 180, 600, 1000。
    aggressiveness : float, 可选
        滤波强度控制参数，范围 [0, 1]。
        0.0: 目标响应位于走廊下限，滤波最强 (最平滑)。
        0.5: 目标响应位于走廊中心，平衡。
        1.0: 目标响应位于走廊上限，滤波最弱 (保留最多细节)。
        默认为 0.0。

    返回:
    -------
    filtered_signal : numpy.ndarray
        滤波后的信号数组。
    """
    # --- 1. 参数计算与验证 ---
    time_filter = np.asarray(time_filter, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    n_samples = len(signal)

    if not (0.0 <= aggressiveness <= 1.0):
        raise ValueError("'aggressiveness' 参数必须在 0.0 到 1.0 之间。")

    if time_filter.shape != signal.shape:
        raise ValueError("时间和信号数组的长度必须一致。")

    sample_rate = 1.0 / np.mean(np.diff(time_filter))

    # --- 2. 定义CFC性能走廊 ---
    corridors = {
        60: (np.array([0.1, 60, 75, 100, 130, 160, 452]), # 
             np.array([0.5, 0.5, -0.3, -1.8, -5.2, -9.2, -40]),
             np.array([-0.5, -1.0, -1.8, -3.8, -8.2, -13.2, -48.3])),
        180: (np.array([0.1, 180, 225, 300, 390, 480, 1310]),
              np.array([0.5, 0.5, -0.3, -1.8, -5.2, -9.2, -40]),
              np.array([-0.5, -1.0, -1.8, -3.8, -8.2, -13.2, -48.3])),
        600: (np.array([0.1, 600, 1000, 1200, 2119, 3865]),
              np.array([0.5, 0.5, 0.5, 0.5, -19.2, -40.0]),
              np.array([-0.5, -1.0, -4.0, -10.4, -30.0, -1e16])), # 用-1e16代替无穷小
        1000: (np.array([0.1, 1000, 1650, 2000, 3496, 6442]),
               np.array([0.5, 0.5, 0.5, 0.5, -18.9, -40.0]),
               np.array([-0.5, -1.0, -4.0, -10.7, -30.0, -1e16]))
    }
    if CFC not in corridors:
        raise ValueError(f"不支持的CFC等级: {CFC}。支持的等级为 {list(corridors.keys())}。")

    # --- 3. 准备FFT与边界填充 ---
    # 确定FFT点数，至少比原始信号长15%
    n_fft = 1 << (int(np.ceil(np.log2(n_samples * 1.15))))
    
    # 计算填充长度
    n_pad = n_fft - n_samples
    n_pre = n_pad // 2
    n_post = n_pad - n_pre

    # 创建填充后的信号数组
    padded_signal = np.zeros(n_fft)

    # 1). 复制原始信号到中间
    padded_signal[n_pre : n_pre + n_samples] = signal

    # 2). 前端对称反射
    # 取 signal[1] 到 signal[n_pre] 的部分，翻转后放到最前面
    front_reflection = signal[1 : n_pre + 1][::-1]
    padded_signal[:n_pre] = front_reflection

    # 3). 后端对称反射
    # 取 signal[-(n_post+1)] 到 signal[-2] 的部分，翻转后放到最后面
    end_reflection = signal[-(n_post + 1) : -1][::-1]
    padded_signal[n_pre + n_samples:] = end_reflection

    # --- 4. 设计滤波器频率响应 ---
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0/sample_rate)
    node_freqs, node_upper_db, node_lower_db = corridors[CFC]

    # 创建上下限的插值函数
    log_node_freqs = np.log10(node_freqs[node_freqs > 0])
    interp_upper = interp1d(log_node_freqs, node_upper_db[node_freqs > 0], kind='linear',
                              bounds_error=False, fill_value=(node_upper_db[0], node_upper_db[-1]))
    interp_lower = interp1d(log_node_freqs, node_lower_db[node_freqs > 0], kind='linear',
                              bounds_error=False, fill_value=(node_lower_db[0], node_lower_db[-1]))

    # 在FFT频率轴上插值得到完整的上下限走廊
    valid_freqs = freq_axis > 0
    log_freq_axis = np.log10(freq_axis[valid_freqs])
    
    upper_db = np.full_like(freq_axis, node_upper_db[0])
    lower_db = np.full_like(freq_axis, node_lower_db[0])
    
    upper_db[valid_freqs] = interp_upper(log_freq_axis)
    lower_db[valid_freqs] = interp_lower(log_freq_axis)

    # 根据 aggressiveness 参数在上下限之间进行线性插值，确定目标响应
    target_db = lower_db + aggressiveness * (upper_db - lower_db)
    
    # 遵循无增益约束
    target_db = np.minimum(target_db, 0)
    filter_gain = 10**(target_db / 20.0)

    # --- 5. 应用滤波器 ---
    signal_fft = np.fft.rfft(padded_signal)
    filtered_fft = signal_fft * filter_gain
    filtered_signal_padded = np.fft.irfft(filtered_fft, n=n_fft)

    # --- 6. 结果提取 ---
    filtered_signal = filtered_signal_padded[n_pre : n_pre + n_samples]

    return filtered_signal

def CFC_filter_SAE_V2(time_filter, signal, CFC=1000, aggressiveness=0.0, taper_ratio=0.1): # 有较大边界伪影
    """
    根据 SAE J211-1 附录C 的规范，使用频域FFT方法对信号进行CFC无相位滤波。
    此版本采用最鲁棒的“去趋势+锥化+零填充”方案，专为处理噪声大、截断的原始信号设计。

    参数:
    ----------
    time_filter : array_like
        时间序列数组，单位为秒(s)。必须是等间隔采样。
    signal : array_like
        需要滤波的信号数组，与 time_filter 对应。
    CFC : int, 可选
        通道频率等级 (Channel Frequency Class)。支持 60, 180, 600, 1000。
        默认为 1000。
    aggressiveness : float, 可选
        滤波强度控制参数，范围 [0, 1]。默认为 0.0 (最强滤波)。
    taper_ratio : float, 可选
        锥化窗应用于信号两端的比例，范围 [0, 1]。
        例如 0.1 代表前后各 5% 的数据被锥化。默认为 0.1。

    返回:
    -------
    filtered_signal : numpy.ndarray
        滤波后的信号数组。
    """

    # --- 1. 参数计算与验证 ---
    time_filter = np.asarray(time_filter, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    n_samples = len(signal)

    if not (0.0 <= aggressiveness <= 1.0):
        raise ValueError("'aggressiveness' 参数必须在 0.0 到 1.0 之间。")
    if not (0.0 <= taper_ratio <= 1.0):
        raise ValueError("'taper_ratio' 参数必须在 0.0 到 1.0 之间。")
    if time_filter.shape != signal.shape:
        raise ValueError("时间和信号数组的长度必须一致。")

    sample_rate = 1.0 / np.mean(np.diff(time_filter))

    # --- 2. 定义CFC性能走廊 ---
    corridors = {
        60: (np.array([0.1, 60, 75, 100, 130, 160, 452]), # 
             np.array([0.5, 0.5, -0.3, -1.8, -5.2, -9.2, -40]),
             np.array([-0.5, -1.0, -1.8, -3.8, -8.2, -13.2, -48.3])),
        180: (np.array([0.1, 180, 225, 300, 390, 480, 1310]),
              np.array([0.5, 0.5, -0.3, -1.8, -5.2, -9.2, -40]),
              np.array([-0.5, -1.0, -1.8, -3.8, -8.2, -13.2, -48.3])),
        600: (np.array([0.1, 600, 1000, 1200, 2119, 3865]),
              np.array([0.5, 0.5, 0.5, 0.5, -19.2, -40.0]),
              np.array([-0.5, -1.0, -4.0, -10.4, -30.0, -1e16])), # 用-1e16代替无穷小
        1000: (np.array([0.1, 1000, 1650, 2000, 3496, 6442]),
               np.array([0.5, 0.5, 0.5, 0.5, -18.9, -40.0]),
               np.array([-0.5, -1.0, -4.0, -10.7, -30.0, -1e16]))
    }
    if CFC not in corridors:
        raise ValueError(f"不支持的CFC等级: {CFC}。支持的等级为 {list(corridors.keys())}。")

    # --- 3. 准备FFT与边界填充 (最鲁棒方案) ---
    n_fft = 1 << (int(np.ceil(np.log2(n_samples * 1.15))))
    
    # 步骤 1: 去趋势
    # 保存趋势，以便最后加回
    trend = np.linspace(signal[0], signal[-1], n_samples)
    detrended_signal = signal - trend
    
    # 步骤 2: 锥化 (Tapering)
    # 创建一个Tukey窗，alpha参数是锥化比例
    taper_window = tukey(n_samples, alpha=taper_ratio)
    tapered_signal = detrended_signal * taper_window
    
    # 步骤 3: 零填充
    padded_signal = np.zeros(n_fft)
    padded_signal[:n_samples] = tapered_signal

    # --- 4. 设计滤波器频率响应 ---
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0/sample_rate)
    node_freqs, node_upper_db, node_lower_db = corridors[CFC]

    # 创建上下限的插值函数
    log_node_freqs = np.log10(node_freqs[node_freqs > 0])
    interp_upper = interp1d(log_node_freqs, node_upper_db[node_freqs > 0], kind='linear',
                              bounds_error=False, fill_value=(node_upper_db[0], node_upper_db[-1]))
    interp_lower = interp1d(log_node_freqs, node_lower_db[node_freqs > 0], kind='linear',
                              bounds_error=False, fill_value=(node_lower_db[0], node_lower_db[-1]))

    # 在FFT频率轴上插值得到完整的上下限走廊
    valid_freqs = freq_axis > 0
    log_freq_axis = np.log10(freq_axis[valid_freqs])
    
    upper_db = np.full_like(freq_axis, node_upper_db[0])
    lower_db = np.full_like(freq_axis, node_lower_db[0])
    
    upper_db[valid_freqs] = interp_upper(log_freq_axis)
    lower_db[valid_freqs] = interp_lower(log_freq_axis)

    # 根据 aggressiveness 参数在上下限之间进行线性插值，确定目标响应
    target_db = lower_db + aggressiveness * (upper_db - lower_db)
    
    # 遵循无增益约束
    target_db = np.minimum(target_db, 0)
    filter_gain = 10**(target_db / 20.0)

    # --- 5. 应用滤波器 ---
    signal_fft = np.fft.rfft(padded_signal)
    filtered_fft = signal_fft * filter_gain
    filtered_signal_padded = np.fft.irfft(filtered_fft, n=n_fft)

    # --- 6. 结果提取与趋势恢复 ---
    # 截取与原始信号等长的部分
    filtered_detrended_signal = filtered_signal_padded[:n_samples]
    
    # 关键一步：将之前移除的趋势加回去
    filtered_signal = filtered_detrended_signal + trend

    return filtered_signal

# ---------------------------------------------------------------------------------------------

# CSV保存函数
def save_csv(data_dict, filename, output_dir=None):
    """优化的CSV保存函数，避免重复DataFrame创建"""
    if output_dir:
        filename = os.path.join(output_dir, filename)
    df = pd.DataFrame(data_dict)
    df.to_csv(filename, sep='\t', index=False, header=False,
              quoting=csv.QUOTE_NONE, quotechar="", escapechar="\t")

for file in path_list:
    case_id = int(file.split('.')[0].split('_')[1])  # 提取case编号
    # if case_id <10:
    #     continue
    print(f'Processing case {case_id}...')

    # 先获取文件大小，一般小于9900KB的xlsx文件会含有nan值，加入bad_case_ids_list
    file_size_kb = os.path.getsize(os.path.join(xls_route, file)) / 1024  # 获取文件大小, 单位KB
    # if file_size_kb < 9900:
    if file_size_kb < 13810:
        print(f"**Warning: {file_size_kb} KB of case {case_id}. Skipping the case.")

    # 只读取需要的列，减少内存占用和读取时间
    usecols = [0, 1, 2, 3, 17, 19, 37, 9, 11, 29]  # 原始列索引
    col_names = ['t_ms', 'x', 'y', 'z', 'vx', 'vy', 'omg_z', 'ax_raw', 'ay_raw', 'ez_raw'] # 为列指定名称
    
    # 1. 创建从原始列索引到目标名称的映射
    col_map = dict(zip(usecols, col_names))
    
    # 2. 读取数据，此时列名为Excel中的原始列名（或索引）
    data = pd.read_excel(os.path.join(xls_route, file), engine='openpyxl', 
                         usecols=usecols, header=None, skiprows=1)
    # 3. 根据映射重命名列
    data = data.rename(columns=col_map)
    
    # 4. 按照我们期望的 col_names 顺序重新排列各列
    data = data[col_names]
    data_values = data.values  # 转换为numpy数组提高访问速度
    
    # 预分配所有需要的变量，现在索引与预期一致
    t_ms = data_values[:, 0]  # ms，用于后续计算
    tim = t_ms * 0.001  # 将ms换算成s
    x, y, z = data_values[:, 1], data_values[:, 2], data_values[:, 3]  # mm
    vx, vy = data_values[:, 4], data_values[:, 5]  # m/s
    omg_z = data_values[:, 6]  # rad/s
    ax_raw, ay_raw, ez_raw = data_values[:, 7], data_values[:, 8], data_values[:, 9]  # m/s^2, rad/s^2

    if np.isnan(vx).any():
        print(f"Warning: NaN Vx values found in case {case_id}. Skipping this case.")
        continue

    # 旧CFC滤波、ISO CFC滤波、ISO CFC Butterworth滤波、SAE J211-1 FFT滤波  输入和流程一样 用循环遍历调用
    CFC_fun_list = [CFC_filter,CFC_filter_ISO_butterworth, CFC_filter_SAE]
    for func in CFC_fun_list:

        func_name = func.__name__
        print(f"Applying {func_name} for case {case_id}...")
        vx_filtered = func(tim, vx, 60) # m/s
        ax = np.gradient(vx_filtered, tim)  # m/s^2，使用np.gradient提高计算速度
        ax_filtered = -func(tim, ax, 180) # 单位：m/s^2，并且取负值方向

        # 对原始加速度直接滤波
        # ax_filtered1 = -func(tim, ax_raw, 60) # 单位：m/s^2，并且取负值方向
        # ax_filtered = func(tim, ax_filtered1, 60) # 单位：m/s^2

        x_name= func_name + '_x' +str(case_id)+'.csv' 
        save_csv({'XI': tim, 'YI': ax_filtered}, x_name, output_path)

        vy_filtered = func(tim, vy, 60)
        ay = np.gradient(vy_filtered, tim)  # m/s^2，使用np.gradient提高计算速度
        ay_filtered = func(tim, ay, 180) # 单位：m/s^2
     
        # 对原始加速度直接滤波
        # ay_filtered1 = func(tim, ay_raw, 60) # 单位：m/s^2
        # ay_filtered = func(tim, ay_filtered1, 60) # 单位：m/s^2

        y_name= func_name + '_y' +str(case_id)+'.csv'
        save_csv({'XI': tim, 'YI': ay_filtered}, y_name, output_path)

        #  Z方向旋转加速度计算向量化处理
        #对应元素点乘np.multiply，对应元素矩阵乘np.dot
        pos = [1638.13, -799.49, -18.65]
        xt = x + pos[0]
        yt = y + pos[1]
        zt = z + pos[2]   #负值是因为重力沉降
        rt = np.sqrt(xt*xt + yt*yt + zt*zt)   #mm 修复矩阵运算错误
        const = np.sqrt(2475**2 + 775**2 + 341**2)   #算MADYMO中的半径值 mm。因为两个原点在x轴上有一定偏移
        temp = omg_z * rt * 0.001   #m/s 使用向量化操作

        temp_filtered = func(tim, temp, 60)

        omg_new = temp_filtered / (const * 0.001)   #rad/s
        rot_acc = np.gradient(omg_new, tim)  # rad/s^2，使用np.gradient提高计算速度
        az = func(tim, rot_acc, 180)
     
        # 对原始加速度直接滤波
        # az1 = func(tim, ez_raw, 60)  # rad/s^2，使用np.gradient提高计算速度
        # az = func(tim, az1, 60)

        z_name = func_name + '_z' + str(case_id) + '.csv'
        save_csv({'XI': tim, 'YI': az}, z_name, output_path)
    
    # LX滤波
    try:
        ax_filtered_LX = -LX_filter(t_ms, vx)  # m/s^2
        x_name_LX='LX_x'+str(case_id)+'.csv' 
        save_csv({'XI': tim, 'YI': ax_filtered_LX}, x_name_LX, output_path)
        
        ay_filtered_LX = LX_filter(t_ms, vy)  # m/s^2
        y_name_LX='LX_y'+str(case_id)+'.csv'
        save_csv({'XI': tim, 'YI': ay_filtered_LX}, y_name_LX, output_path)

        #  Z方向旋转加速度计算向量化处理
        avg_dt = np.mean(np.diff(t_ms))    # 单位 ms
        fs = 1000 / avg_dt if avg_dt > 0 else 1000  # 采样频率
        cutoff = 300  # 截止频率

        pos = [1638.13, -799.49, -18.65]
        xt = x + pos[0]
        yt = y + pos[1]
        zt = z + pos[2]   #负值是因为重力沉降
        rt = np.sqrt(xt*xt + yt*yt + zt*zt)   #mm 修复矩阵运算错误
        const = np.sqrt(2475**2 + 775**2 + 341**2)   #算MADYMO中的半径值 mm。因为两个原点在x轴上有一定偏移
        temp = omg_z * rt * 0.001   #m/s 使用向量化操作

        temp_filtered_LX = LX_butter_lowpass_filter(temp, cutoff, fs)

        omg_new_LX = temp_filtered_LX / (const * 0.001)   #rad/s
        rot_acc_LX = np.gradient(omg_new_LX, tim)  # rad/s^2，使用np.gradient提高计算速度
        az_LX = LX_butter_lowpass_filter(rot_acc_LX, cutoff/2, fs)

        z_name_LX = 'LX_z' + str(case_id) + '.csv'
        save_csv({'XI': tim, 'YI': az_LX}, z_name_LX, output_path)

        print(f"Case {case_id} processed successfully with LX filter.")

    except Exception as e:
        print(f"LX  Error processing case {case_id}: {str(e)}")
        continue
