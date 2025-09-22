# 二次检查，筛选不好的数据
# 增加Y和Z方向的检查；增加150ms是否跑完的检查
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

x_csv_data_dir = r'I:\000 LX\dataset0715\03\acc_data_0918'
params_path = r'I:\000 LX\dataset0715\03\distribution_0919.csv'
save_dir = './异常数据_0918'
# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

def save_xyz_acc_plots(time, ax, ay, az, params, save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # X方向子图
    axes[0].plot(time, ax, label='Ax ($m/s^2$)', color='blue')
    axes[0].set_ylabel('Ax ($m/s^2$)')
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].legend()
    axes[0].grid()
    
    # Y方向子图
    axes[1].plot(time, ay, label='Ay ($m/s^2$)', color='green')
    axes[1].set_ylabel('Ay ($m/s^2$)')
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].legend()
    axes[1].grid()
    
    # Z方向子图
    axes[2].plot(time, az, label='Az ($rad/s^2$)', color='red')
    axes[2].set_ylabel('Az ($rad/s^2$)')
    axes[2].set_xlabel('Time (s)')
    axes[2].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[2].legend()
    axes[2].grid()
    
    # 标题包含速度、角度、重叠率等信息，前两者保留一位小数，重叠率用百分比表示，保留整数
    title_str = f"Case {params['case_id']} - velocity: {params['speed']:.1f} km/h, angle: {params['angle']:.1f}°, overlap: {params['overlap'] * 100:.0f}%"
    if params['anomaly_conditions']:
        # 打印异常条件及其对应的值，值保留两位小数；如果是比例则保留百分比，保留一位小数
        anomaly_strs = []
        for cond, val in zip(params['anomaly_conditions'], params['anomaly_values']):
            if '比例' in cond:
                anomaly_strs.append(f"{cond} ({val * 100:.1f}%)")
            else:
                anomaly_strs.append(f"{cond} ({val:.2f})")
        title_str += "\nAnomalies: " + "; ".join(anomaly_strs)

    fig.suptitle(title_str, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
        #print(f"Case {params['case_id']} 保存图像到 {save_path}")
    else:
        plt.show()

try:
    if params_path.endswith('.csv'):
        data = pd.read_csv(params_path)
    elif params_path.endswith('.npz'):
        data = np.load(params_path, allow_pickle=True)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .npz files.")
    # 转为df
    df = pd.DataFrame({
        'case_id': data['case_id'].astype(int),
        'have_run': data['have_run'].astype(bool),
        'impact_velocity': data['impact_velocity'].astype(np.float64), #km/h,~25~65
        'impact_angle': data['impact_angle'].astype(np.float64), # 角度，单位为度
        'overlap': data['overlap'].astype(np.float64) # 重叠率，单位为百分比
    }).set_index('case_id')
    # velocity_np = data['impact_velocity'].astype(np.float64) #km/h,~25~65
    # angle_np = data['impact_angle'].astype(np.float64) # 角度，单位为度
    # overlap_np = data['overlap'].astype(np.float64) #
    print(f"成功加载 '{params_path}' 文件，其中共找到 {len(df)} 个样本。")
except FileNotFoundError:
    print(f"错误：无法找到参数文件 '{params_path}'。请检查文件路径。")


x_csv_path_list = []
y_csv_path_list = []
z_csv_path_list = []
# 遍历以x开头，.csv结尾的文件，同时查找y和z文件
if not os.path.exists(x_csv_data_dir):
    print(f"目录 {x_csv_data_dir} 不存在，请检查路径。")
for root, dirs, files in os.walk(x_csv_data_dir):
    for file in files:
        if file.startswith('x') and file.endswith('.csv'):
            x_csv_path_list.append(os.path.join(root, file))
        elif file.startswith('y') and file.endswith('.csv'):
            y_csv_path_list.append(os.path.join(root, file))
        elif file.startswith('z') and file.endswith('.csv'):
            z_csv_path_list.append(os.path.join(root, file))
print(f"在{x_csv_data_dir}找到 {len(x_csv_path_list)} 个x文件，{len(y_csv_path_list)} 个y文件，{len(z_csv_path_list)} 个z文件。")

# x和.csv之间的数字是case的编号，按case编号排序。索引从1开始。
# 例如：x1.csv, x2.csv, ..., x10.csv
x_csv_path_list.sort(key=lambda x: int(x.split(os.sep)[-1].split('.')[0][1:]))
y_csv_path_list.sort(key=lambda x: int(x.split(os.sep)[-1].split('.')[0][1:]))
z_csv_path_list.sort(key=lambda x: int(x.split(os.sep)[-1].split('.')[0][1:]))
# 这些case的速度范围为25km/h到65km/h，部分在25km/h以下。按照速度范围进行分组（25km/h以下，25-35km/h，35-45km/h，45-55km/h，55-65km/h）
speed_path_groups = {
    '25kmh以下': [],
    '25-35kmh': [],
    '35-45kmh': [],
    '45-55kmh': [],
    '55-65kmh': []
}
for i, x_csv_path in enumerate(x_csv_path_list):
    # 获取case编号
    case_id = int(x_csv_path.split(os.sep)[-1].split('.')[0][1:])
    # 获取对应的速度
    speed = df.at[case_id, 'impact_velocity']
    if speed < 25:
        speed_path_groups['25kmh以下'].append(x_csv_path)
    elif 25 <= speed < 35:
        speed_path_groups['25-35kmh'].append(x_csv_path)
    elif 35 <= speed < 45:
        speed_path_groups['35-45kmh'].append(x_csv_path)
    elif 45 <= speed < 55:
        speed_path_groups['45-55kmh'].append(x_csv_path)
    elif 55 <= speed <= 65:
        speed_path_groups['55-65kmh'].append(x_csv_path)

# 开始遍历检查可能为异常的Ax数据，并将疑似数据画图保存以便人工筛查
has_nan_case_ids = []
anormal_case_ids = []
has_not_runall_case_ids = []
def check_xyz(range_name, 
              max_threshold_upper=[100.0, 100.0, 100.0, 100.0, 100.0],
              min_threshold_upper=[-60.0, -60.0, -60.0, -60.0, -60.0], 
              min_threshold_lower=[-300.0, -300.0, -300.0, -300.0, -300.0],
              ax_avg_threshold=[-10.0, -10.0, -10.0, -10.0, -10.0],
              y_z_abs_max_threshold=[100.0, 100.0, 100.0, 100.0, 100.0],

              half_over_ratio_threshold=0.8, 
              zero_nums_ratio_threshold=0.15, zero_range_threshold=1,
              save_dir='./异常数据'):
    
    # 获取当前速度范围对应的索引
    range_names = ['25kmh以下', '25-35kmh', '35-45kmh', '45-55kmh', '55-65kmh']
    range_idx = range_names.index(range_name)
    
    # 获取当前范围对应的阈值参数
    current_max_threshold_upper = max_threshold_upper[range_idx]
    current_min_threshold_upper = min_threshold_upper[range_idx]
    current_min_threshold_lower = min_threshold_lower[range_idx]
    current_ax_avg_threshold = ax_avg_threshold[range_idx]
    current_y_z_abs_max_threshold = y_z_abs_max_threshold[range_idx]

    global has_nan_case_ids, anormal_case_ids, has_not_runall_case_ids

    print(f"检查 {range_name} 的XYZ数据：")
    print(f"使用阈值 - max_upper: {current_max_threshold_upper}, min_upper: {current_min_threshold_upper}, min_lower: {current_min_threshold_lower}, ax_avg: {current_ax_avg_threshold}, yz_abs_max: {current_y_z_abs_max_threshold}")
    
    for x_csv_path in speed_path_groups[range_name]:
        # 单位为m/s²，且主要为负值的float类型
        case_id = int(x_csv_path.split(os.sep)[-1].split('.')[0][1:])
        
        # 构建对应的y和z文件路径
        y_csv_path = x_csv_path.replace('x', 'y')
        z_csv_path = x_csv_path.replace('x', 'z')
        
        # 检查y和z文件是否存在
        if not os.path.exists(y_csv_path) or not os.path.exists(z_csv_path):
            print(f"Case {case_id} 缺少Y或Z文件，跳过。")
            continue
            
        params = {
            'case_id': case_id,
            'speed': df.at[case_id, 'impact_velocity'],
            'angle': df.at[case_id, 'impact_angle'], 
            'overlap': df.at[case_id, 'overlap'],
            'anomaly_conditions': [],
            'anomaly_values': []
        }
        
        # 读取X、Y、Z数据和时间
        time = pd.read_csv(x_csv_path, sep='\t', header=None, usecols=[0], dtype=np.float32).to_numpy().flatten()
        ax_full = pd.read_csv(x_csv_path, sep='\t', header=None, usecols=[1], dtype=np.float32).to_numpy().flatten()
        ay_full = pd.read_csv(y_csv_path, sep='\t', header=None, usecols=[1], dtype=np.float32).to_numpy().flatten()
        az_full = pd.read_csv(z_csv_path, sep='\t', header=None, usecols=[1], dtype=np.float32).to_numpy().flatten()
        
        # 检查NaN值
        if np.isnan(ax_full).any() or np.isnan(ay_full).any() or np.isnan(az_full).any():
            has_nan_case_ids.append(case_id)
            print(f"****!!!Warning: Case {case_id} 的数据中包含 NaN 值。")
            continue
            
        # X方向异常检查
        ax_max = np.max(ax_full)
        ax_min = np.min(ax_full)
        ax_avg = np.mean(ax_full)
        # 0附近值的个数所占比例
        zero_nums = np.sum(np.abs(ax_full) < zero_range_threshold)
        zero_ratio = zero_nums / len(ax_full) if len(ax_full) > 0 else np.inf
        # 后半段均值与前半段均值的比值
        half_length = len(ax_full) // 2
        half_avg_behind = np.mean(ax_full[half_length:])
        half_avg_front = np.mean(ax_full[:half_length])
        half_avg_ratio = half_avg_behind / half_avg_front if half_avg_front != 0 else np.inf
        # 150ms附近的值的绝对值的平均值
        near_150ms_avg = np.mean(np.abs(ax_full[(time >= 0.145) & (time <= 0.150)])) if len(ax_full) > 0 else np.nan

        # X方向异常条件
        x_anomaly_condition = {
            'X最大值超过阈值': ax_max > current_max_threshold_upper,
            'X最小值超过阈值': ax_min < current_min_threshold_lower or ax_min > current_min_threshold_upper, # 最小值就是X方向波形的峰值，前者避免其绝对值过大，后者条件避免过小
            'X平均值超过阈值': ax_avg > current_ax_avg_threshold,
            'X0附近值比例过大': zero_ratio > zero_nums_ratio_threshold,
            # 'X前半段均值超过阈值': half_avg_front > current_ax_avg_threshold / 2,
            # 'X后半段均值与前半段均值的比值过大': abs(half_avg_ratio) > half_over_ratio_threshold,
            '150ms附近的值的绝对值的平均值过大': near_150ms_avg > np.abs(ax_min) / 4 # 存在该异常条件的caseid后面要打印出来，视为not_runall
        }
        #x_condition_values = [ax_max, ax_min, ax_avg, zero_ratio, half_avg_front, half_avg_ratio, near_150ms_avg]
        x_condition_values = [ax_max, ax_min, ax_avg, zero_ratio, near_150ms_avg]
        # Y和Z方向异常检查（只检查绝对值最大值）
        ay_abs_max = np.max(np.abs(ay_full))
        az_abs_max = np.max(np.abs(az_full))
        
        y_z_anomaly_condition = {
            'Y绝对值最大值超过阈值': ay_abs_max > current_y_z_abs_max_threshold,
            'Z绝对值最大值超过阈值': az_abs_max > current_y_z_abs_max_threshold
        }
        y_z_condition_values = [ay_abs_max, az_abs_max]
        
        # 合并所有异常条件
        all_anomaly_conditions = {**x_anomaly_condition, **y_z_anomaly_condition}
        all_condition_values = x_condition_values + y_z_condition_values
        
        if any(all_anomaly_conditions.values()): # 检查是否有异常条件触发
            anormal_case_ids.append(case_id)
            print(f"Case {case_id} 的数据可能存在异常，异常的条件有：")
            for i, (cond_name, cond) in enumerate(all_anomaly_conditions.items()):
                if cond:
                    print(f" - 条件 {i + 1} 触发: {cond_name}")
                    params['anomaly_conditions'].append(cond_name)
                    params['anomaly_values'].append(all_condition_values[i])
                    # 如果是150ms附近的值的绝对值的平均值过大，则视为not_runall
                    if cond_name == '150ms附近的值的绝对值的平均值过大':
                        has_not_runall_case_ids.append(case_id)

            # 画图保存
            save_xyz_acc_plots(time, ax_full, ay_full, az_full, params, 
                             save_path=os.path.join(save_dir, f"case_{case_id}_xyz_plot.png") if save_dir else None)
            
    # 打印分隔线
    print("=" * 50)
    print(f"{range_name} 的XYZ数据检查完毕。\n")
    # 打印这部分数据的总数及可能异常数据的数量
    total_cases = len(speed_path_groups[range_name])
    print(f"{range_name} 总共有 {total_cases} 个案例。")
    print(f"可能存在异常的案例累计有 {len(anormal_case_ids)} 个。\n")
    # not_runall的case_id都打印出来
    if has_not_runall_case_ids:
        has_not_runall_case_ids = list(set(has_not_runall_case_ids)) # 去重
        has_not_runall_case_ids.sort()
        print(f"以下 {len(has_not_runall_case_ids)} 个案例可能未跑完（150ms附近的值的绝对值的平均值过大）：")
        print(has_not_runall_case_ids)
    # nan的case_id都打印出来
    if has_nan_case_ids:
        has_nan_case_ids = list(set(has_nan_case_ids)) # 去重
        has_nan_case_ids.sort()
        print(f"以下 {len(has_nan_case_ids)} 个案例的数据中包含 NaN 值：")
        print(has_nan_case_ids)

    print("=" * 50)

if __name__ == "__main__":

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for range_name in speed_path_groups.keys():
        check_xyz(range_name=range_name, 
                max_threshold_upper=[40.0, 65.0, 85.0, 105.0, 125.0], 
                min_threshold_lower=[-220.0, -290.0, -370.0, -450.0, -550.0], 
                min_threshold_upper=[-30.0, -40.0, -50.0, -60.0, -70.0], 
                ax_avg_threshold=[-15.0, -20.0, -25.0, -32.0, -40.0],
                y_z_abs_max_threshold=[80.0, 105.0, 140.0, 170.0, 200.0],
                half_over_ratio_threshold=0.8, 
                zero_nums_ratio_threshold=0.15, zero_range_threshold=1,
                save_dir=save_dir)