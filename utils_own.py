# -*- coding: utf-8 -*-
'''
杂七杂八的操作，如操作数据文件，画图等
'''
# %% 删除指定caseid list的csv文件:x{xxx}.csv / y{xxxx}.csv / z{xxx}.csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 指定要删除的case id列表
# 该csv只有一列无表头，且数据类型为整数
# case_ids_to_delete = pd.read_csv(r'异常_addition600.csv', header=None).iloc[:, 0].astype(int).tolist()

# case_ids_to_delete = [829, 838, 895, 896, 1021 ,1062,1079,1130,1183,1249,1267,1331,1374,1486,1504,1543,1712,1778] # I:\000 LX\dataset0715\03\acc_data_150ms_0911

case_ids_to_delete =  [1811, 2046,2063,2345] # I:\000 LX\dataset0715\03\acc_data_150ms_0914_NEW600

file_path = r'I:\000 LX\dataset0715\03\acc_data_150ms_0914_NEW600'  
delete_count = 0
for case_id in case_ids_to_delete:
    for axis in ['x', 'y', 'z']:
        file_name = f'{axis}{case_id}.csv'
        full_path = os.path.join(file_path, file_name)
        if os.path.exists(full_path):
            os.remove(full_path)
            print(f"Deleted file: {full_path}")
            delete_count += 1
        else:
            print(f"File not found, could not delete: {full_path}")
print(f"Total files deleted: {delete_count}")


# %% 根据csv文件夹下的caseid，绘制三个方向加速度波形图
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_acceleration_waveforms(case_ids, save_plots=False, show_plots=True, data_dir=None, params_df=None, save_dir='./'):
    """
    绘制指定case的三个方向加速度波形
    
    Parameters:
    case_ids: list, 要绘制的case编号列表
    save_plots: bool, 是否保存图片
    show_plots: bool, 是否显示图片
    data_dir: str, CSV文件所在路径
    params: dict, 工况参数
    save_dir: str, 图片保存路径，默认当前路径
    """
    print(f"Plotting acceleration waveforms for cases: {case_ids}, from {data_dir}")
    for case_id in case_ids:
        try:
            # 读取三个方向的CSV文件
            x_file = os.path.join(data_dir, f'x{case_id}.csv')
            y_file = os.path.join(data_dir, f'y{case_id}.csv')
            z_file = os.path.join(data_dir, f'z{case_id}.csv')

            # 检查文件是否存在
            if not all(os.path.exists(f) for f in [x_file, y_file, z_file]):
                # print(f"Warning: CSV files for case {case_id} not found in {data_dir}. Skipping...")
                continue
            
            # 读取数据
            x_data = pd.read_csv(x_file, sep='\t', header=None, names=['time', 'ax'])
            y_data = pd.read_csv(y_file, sep='\t', header=None, names=['time', 'ay'])
            z_data = pd.read_csv(z_file, sep='\t', header=None, names=['time', 'az'])

            params_row = params_df.loc[case_id]
            params = { 
                'velocity': float(params_row['impact_velocity']),
                'angle': float(params_row['impact_angle']),
                'overlap': float(params_row['overlap'])
            }
            # 创建子图
            fig, axes = plt.subplots(3, 1, figsize=(12, 12))
            title_str = f'Case {case_id} \nVelocity: {params["velocity"]:.1f}km/h Angle: {params["angle"]:.1f}° Overlap: {params["overlap"]*100:.1f}%'
            fig.suptitle(title_str, fontsize=14, fontweight='bold')
            
            # ********************************************************************
            # 只选择time = 2ms,4ms,....150ms的数据点进行绘图;
            time_need = np.round(np.arange(0.001, 0.151, 0.001), 8) # 需要的时间点，单位秒
            x_data['time_rounded'] = np.round(x_data['time'], 8)
            time_mask = x_data['time_rounded'].isin(time_need)
            x_data = x_data[time_mask]
            y_data = y_data[time_mask]
            z_data = z_data[time_mask] 
            # print(len(x_data), len(y_data), len(z_data))
            if len(time_need) != len(x_data) or len(time_need) != len(y_data) or len(time_need) != len(z_data):
                print(f"***Warning: Case {case_id} does not have the required time points. Skipping...")
            # ********************************************************************

            # X方向加速度
            axes[0].plot(x_data['time'] * 1000, x_data['ax'], 'b-', linewidth=1.5, label='X-direction')
            axes[0].set_ylabel('Acceleration (m/s²)', fontsize=12)
            axes[0].set_title('X-direction Acceleration', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Y方向加速度
            axes[1].plot(y_data['time'] * 1000, y_data['ay'], 'r-', linewidth=1.5, label='Y-direction')
            axes[1].set_ylabel('Acceleration (m/s²)', fontsize=12)
            axes[1].set_title('Y-direction Acceleration', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
            # Z方向旋转加速度
            axes[2].plot(z_data['time'] * 1000, z_data['az'], 'g-', linewidth=1.5, label='Z-direction (Rotational)')
            axes[2].set_xlabel('Time (ms)', fontsize=12)
            axes[2].set_ylabel('Angular Acceleration (rad/s²)', fontsize=12)
            axes[2].set_title('Z-direction Rotational Acceleration', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()
 
            # # --------------设置子图y轴显示范围--------------
            # axes[0].set_ylim(-650, 250)  # X方向加速度范围
            # axes[1].set_ylim(-300, 300)  # Y方向加速度
            # axes[2].set_ylim(-300, 300)  # Z方向旋转加速度
            # # ---------------------------------------------
         
            # 调整子图间距
            plt.tight_layout()
            
            # 保存图片
            if save_plots:
                plot_filename = f'case_{case_id}_crash_pulse.png'
                plt.savefig(os.path.join(save_dir, plot_filename), dpi=300, bbox_inches='tight')
                print(f"Plot saved as {plot_filename}")
            
            # 显示图片
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error plotting case {case_id}: {str(e)}")


# 加载工况参数
case_params_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_0917.csv'
if not os.path.exists(case_params_path):
    print(f"Error: Case parameters file not found at {case_params_path}")
    exit()
with open(case_params_path, 'r') as f:
    if case_params_path.endswith('.csv'):
        all_case_params = pd.read_csv(case_params_path)
        params_df = all_case_params.set_index('case_id')
    else:
        all_case_params = np.load(case_params_path, allow_pickle=True)
        params_df = pd.DataFrame({
            key: all_case_params[key]
            for key in all_case_params.files
        }).set_index('case_id')

example_case_ids = list(np.arange(1, 10))  # 示例 case_id 列表
data_dir = r'E:\课题组相关\理想项目\仿真数据库相关\acc_data_before0915'
save_dir = r'E:\课题组相关\理想项目\仿真数据库相关\acc_data_before0915\acceleration_plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plot_acceleration_waveforms(example_case_ids, save_plots=True, show_plots=False, data_dir=data_dir, params_df=params_df, save_dir=save_dir)

# %% 筛选波形case 打包为.npz文件数据集
# 1. 碰撞工况参数：impact_angle 在[-45°, 45°]之间; overlap的绝对值>0.15
# 2. 碰撞波形：x方向碰撞波形的加速度值在[-500, 100]之间; y方向和z方向碰撞波形的加速度值在[-150, 150]之间
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def package_pulse_data(pulse_dir, params_path, case_id_list, output_path, downsample_indices=None):
    """
    处理、降采样并将指定案例的输入参数和输出波形数据打包在一起。

    该函数会读取工况参数文件，并根据给定的 case_id 列表，匹配对应的
    原始波形CSV文件。然后将输入参数、输出波形和 case_id 作为一个整体
    保存到一个结构化的 .npz 文件中。

    :param pulse_dir: 存放原始波形CSV文件的目录。
    :param params_path: 包含所有工况参数的 .npz 文件路径 (包含 'case_id' 列)。
    :param case_id_list: 需要处理的案例ID列表。
    :param output_path: 打包后的 .npz 文件保存路径。
    :param downsample_indices: 用于降采样的索引数组。如果为None，则默认抽取200个点。
    """
    if downsample_indices is None:
        downsample_indices = np.arange(100, 20001, 100)

    # --- 1. 加载并索引工况参数 ---
    try:
        all_params_data = np.load(params_path)
        # 使用 pandas DataFrame以便于通过 case_id 高效查找
        params_df = pd.DataFrame({
            'case_id': all_params_data['case_id'],
            'impact_velocity': all_params_data['impact_velocity'],
            'impact_angle': all_params_data['impact_angle'],
            'overlap': all_params_data['overlap']
        }).set_index('case_id')
    except Exception as e:
        print(f"错误：加载或处理工况参数文件 '{params_path}' 时出错: {e}")
        return

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 用于存储最终数据的列表
    processed_case_ids = []
    processed_params = []
    processed_waveforms = []

    print(f"开始处理 {len(case_id_list)} 个案例，将输入和输出打包在一起...")
    for case_id in tqdm(case_id_list, desc="Packaging pulse Data"):
        try:
            # --- 2. 确认参数存在 ---
            if case_id not in params_df.index:
                print(f"警告：在参数文件中未找到案例 {case_id}，已跳过。")
                continue

            # --- 3. 读取并处理波形 ---
            x_path = os.path.join(pulse_dir, f'x{case_id}.csv')
            y_path = os.path.join(pulse_dir, f'y{case_id}.csv')
            z_path = os.path.join(pulse_dir, f'z{case_id}.csv')

            if not all(os.path.exists(p) for p in [x_path, y_path, z_path]):
                print(f"警告：案例 {case_id} 的波形文件不完整，已跳过。")
                continue

            ax_full = pd.read_csv(x_path, sep='\t', header=None, usecols=[1]).values
            ay_full = pd.read_csv(y_path, sep='\t', header=None, usecols=[1]).values
            az_full = pd.read_csv(z_path, sep='\t', header=None, usecols=[1]).values

            ax_sampled = ax_full[downsample_indices]
            ay_sampled = ay_full[downsample_indices]
            az_sampled = az_full[downsample_indices]
            
            waveforms_np = np.stack([ax_sampled, ay_sampled, az_sampled]).squeeze() # 形状 (3, 200)

            # --- 4. 提取匹配的参数 ---
            params_row = params_df.loc[case_id]
            params_np = np.array([
                params_row['impact_velocity'],
                params_row['impact_angle'],
                params_row['overlap']
            ], dtype=np.float32) # 形状 (3,)

            # --- 5. 添加到结果列表 ---
            processed_case_ids.append(case_id)
            processed_params.append(params_np)
            processed_waveforms.append(waveforms_np)

        except Exception as e:
            print(f"警告：处理案例 {case_id} 时发生错误 '{e}'，已跳过。")
            continue
            
    if not processed_case_ids:
        print("错误：没有成功处理任何数据，未生成输出文件。")
        return

    # --- 6. 将数据列表转换为Numpy数组并保存 ---
    final_case_ids = np.array(processed_case_ids, dtype=int) # 形状 (N,)
    final_params = np.stack(processed_params, axis=0) # 形状 (N, 3)
    final_waveforms = np.stack(processed_waveforms, axis=0) # 形状 (N, 3, 200)

    np.savez(
        output_path,
        case_ids=final_case_ids,
        params=final_params,
        waveforms=final_waveforms
    )
    print(f"数据打包完成，已保存至 {output_path}")
    print(f"成功处理并打包的数据数目：{len(final_case_ids)}")
    print(f"打包后文件内容: case_ids shape={final_case_ids.shape}, params shape={final_params.shape}, waveforms shape={final_waveforms.shape}")

# 保留package_pulse_data输出的数据包中符号两方面条件的数据：
# 1. 碰撞工况参数：impact_angle 在[-45°, 45°]之间; overlap的绝对值>0.15
# 2. 碰撞波形：x方向碰撞波形的加速度值在[-500, 100]之间; y方向和z方向碰撞波形的加速度值在[-150, 150]之间
def filter_and_save_data(input_path, output_path):
    try:
        with np.load(input_path) as data:
            filtered_params = data['params']
            filtered_waveforms = data['waveforms']
            filtered_case_ids = data['case_ids']
            # 应用过滤条件
            valid_indices = np.where(
                # 参数条件
                (filtered_params[:, 1] >= -45) & (filtered_params[:, 1] <= 45) & (np.abs(filtered_params[:, 2]) > 0.15) &
                # 波形条件 - 使用np.all确保所有时间点都满足条件
                np.all((filtered_waveforms[:, 0, :] >= -500) & (filtered_waveforms[:, 0, :] <= 100), axis=1) &
                np.all((filtered_waveforms[:, 1, :] >= -150) & (filtered_waveforms[:, 1, :] <= 150), axis=1) &
                np.all((filtered_waveforms[:, 2, :] >= -150) & (filtered_waveforms[:, 2, :] <= 150), axis=1)
            )[0]

            # 保存过滤后的数据
            np.savez(
                output_path,
                case_ids=filtered_case_ids[valid_indices],
                params=filtered_params[valid_indices],
                waveforms=filtered_waveforms[valid_indices]
            )
            print(f"过滤后的数据已保存至 {output_path}")
            print(f"过滤后数据数目：{len(valid_indices)}")
    except Exception as e:
        print(f"错误：处理文件 '{input_path}' 时出错: {e}")

if __name__ == '__main__':
    pulse_dir = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\acceleration_data_all1800'
    params_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution_0825_final.npz'
    output_dir = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关'
    
    # 分割训练集和测试集
    all_case_ids = np.load(params_path)['case_id']
    np.random.shuffle(all_case_ids)
    num_train = int(len(all_case_ids) / 6 * 5)
    train_case_ids = all_case_ids[:num_train]
    test_case_ids = all_case_ids[num_train:]


    # print("\n打包训练集数据...")
    # package_pulse_data(
    #     pulse_dir=pulse_dir,
    #     params_path=params_path,
    #     case_id_list=train_case_ids,
    #     output_path=os.path.join(output_dir, 'packaged_data_train.npz')
    # )
    filter_and_save_data(
        input_path=os.path.join(output_dir, 'packaged_data_train.npz'),
        output_path=os.path.join(output_dir, 'filtered_data_train.npz')
    )

    # print("\n打包测试集数据...")
    # package_pulse_data(
    #     pulse_dir=pulse_dir,
    #     params_path=params_path,
    #     case_id_list=test_case_ids,
    #     output_path=os.path.join(output_dir, 'packaged_data_test.npz')
    # )
    filter_and_save_data(
        input_path=os.path.join(output_dir, 'packaged_data_test.npz'),
        output_path=os.path.join(output_dir, 'filtered_data_test.npz')
    )


# %% 画图对比不同滤波方法
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['mathtext.default'] = 'regular'  # 使用常规字体渲染数学文本
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']  # 添加备用字体
plt.rcParams['axes.unicode_minus'] = False
def plot_filtered_comparison(case_ids, filter_methods=['CFC_filter', 'CFC_filter_ISO', 'CFC_filter_ISO_butterworth', 'CFC_filter_SAE', 'LX'], 
                           save_plots=False, show_plots=True, data_path=None, plots_save_dir='./'):
    """
    绘制指定case的不同滤波方法比较图
    
    Parameters:
    case_ids: list, 要绘制的case编号列表
    filter_methods: list, 要比较的滤波方法列表
    save_plots: bool, 是否保存图片
    show_plots: bool, 是否显示图片
    data_path: str, CSV文件所在路径，默认为当前路径
    """
    if data_path is None:
        save_plots = False
        print("Warning: data_path is None, cannot save plots.")
    # 定义颜色和线型
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    linestyles = [
    '-',           # 实线
    '--',          # 虚线
    '-.',          # 点划线
    ':',           # 点线
    (0, (5, 1)),   # 长虚线
    (0, (3, 1, 1, 1)),  # 复杂虚线
    (0, (1, 1)),   # 密点线
    (0, (5, 5)),   # 均匀虚线
    (0, (3, 5, 1, 5)),  # 不规则虚线
    'solid'        # 实线（完整写法）
    ]
    
    for case_id in case_ids:
        try:
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            fig.suptitle(f'Case {case_id} - Filtered Acceleration Comparison', fontsize=16, fontweight='bold')
            
            # 为每个滤波方法绘制波形
            for i, method in enumerate(filter_methods):
                color = colors[i % len(colors)]
                linestyle = linestyles[i % len(linestyles)]
                
                # 构建文件名
                x_file = os.path.join(data_path, f'{method}_x{case_id}.csv')
                y_file = os.path.join(data_path, f'{method}_y{case_id}.csv')
                z_file = os.path.join(data_path, f'{method}_z{case_id}.csv')
                
                # 检查文件是否存在
                if not all(os.path.exists(f) for f in [x_file, y_file, z_file]):
                    print(f"Warning: {method} files for case {case_id} not found. Skipping this method...")
                    continue
                
                # 读取数据
                x_data = pd.read_csv(x_file, sep='\t', header=None, names=['time', 'ax'])
                y_data = pd.read_csv(y_file, sep='\t', header=None, names=['time', 'ay'])
                z_data = pd.read_csv(z_file, sep='\t', header=None, names=['time', 'az'])
                
                # X方向加速度
                axes[0].plot(x_data['time'] * 1000, x_data['ax'], color=color, linestyle=linestyle,
                           linewidth=1.5, label=f'{method} X-direction', alpha=0.8)
                
                # Y方向加速度
                axes[1].plot(y_data['time'] * 1000, y_data['ay'], color=color, linestyle=linestyle,
                           linewidth=1.5, label=f'{method} Y-direction', alpha=0.8)
                
                # Z方向旋转加速度
                axes[2].plot(z_data['time'] * 1000, z_data['az'], color=color, linestyle=linestyle,
                           linewidth=1.5, label=f'{method} Z-direction', alpha=0.8)
            
            # 设置子图属性
            axes[0].set_ylabel('Acceleration (m/s²)', fontsize=12)
            axes[0].set_title('X-direction Acceleration Comparison', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            axes[1].set_ylabel('Acceleration (m/s²)', fontsize=12)
            axes[1].set_title('Y-direction Acceleration Comparison', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            axes[2].set_xlabel('Time (ms)', fontsize=12)
            axes[2].set_ylabel('Angular Acceleration (rad/s²)', fontsize=12)
            axes[2].set_title('Z-direction Rotational Acceleration Comparison', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            if save_plots:
                plot_filename = f'case_{case_id}_filter_comparison.png'
                plt.savefig(os.path.join(plots_save_dir, plot_filename), dpi=300, bbox_inches='tight')
                print(f"Comparison plot saved as {os.path.join(plots_save_dir, plot_filename)}")
            
            # 显示图片
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error plotting comparison for case {case_id}: {str(e)}")

def plot_single_filter_waveforms(case_ids, filter_method='CFC_filter', 
                                save_plots=False, show_plots=True, data_path=None, plots_save_dir='./'):
    """
    绘制指定滤波方法的三个方向加速度波形
    
    Parameters:
    case_ids: list, 要绘制的case编号列表
    filter_method: str, 滤波方法名称
    save_plots: bool, 是否保存图片
    show_plots: bool, 是否显示图片
    data_path: str, CSV文件所在路径，默认为当前路径
    """
    if data_path is None:
        save_plots = False
        print("Warning: data_path is None, cannot save plots.")

    for case_id in case_ids:
        try:
            # 构建文件名
            x_file = os.path.join(data_path, f'{filter_method}_x{case_id}.csv')
            y_file = os.path.join(data_path, f'{filter_method}_y{case_id}.csv')
            z_file = os.path.join(data_path, f'{filter_method}_z{case_id}.csv')
            
            # 检查文件是否存在
            if not all(os.path.exists(f) for f in [x_file, y_file, z_file]):
                print(f"Warning: {filter_method} files for case {case_id} not found. Skipping...")
                continue
            
            # 读取数据
            x_data = pd.read_csv(x_file, sep='\t', header=None, names=['time', 'ax'])
            y_data = pd.read_csv(y_file, sep='\t', header=None, names=['time', 'ay'])
            z_data = pd.read_csv(z_file, sep='\t', header=None, names=['time', 'az'])
            
            # 创建子图
            fig, axes = plt.subplots(3, 1, figsize=(12, 12))
            fig.suptitle(f'Case {case_id} - {filter_method} Filtered Waveforms', fontsize=16, fontweight='bold')
            
            # X方向加速度
            axes[0].plot(x_data['time'] * 1000, x_data['ax'], 'b-', linewidth=1.5, label='X-direction')
            axes[0].set_ylabel('Acceleration (m/s²)', fontsize=12)
            axes[0].set_title(f'X-direction Acceleration ({filter_method})', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Y方向加速度
            axes[1].plot(y_data['time'] * 1000, y_data['ay'], 'r-', linewidth=1.5, label='Y-direction')
            axes[1].set_ylabel('Acceleration (m/s²)', fontsize=12)
            axes[1].set_title(f'Y-direction Acceleration ({filter_method})', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
            # Z方向旋转加速度
            axes[2].plot(z_data['time'] * 1000, z_data['az'], 'g-', linewidth=1.5, label='Z-direction (Rotational)')
            axes[2].set_xlabel('Time (ms)', fontsize=12)
            axes[2].set_ylabel('Angular Acceleration (rad/s²)', fontsize=12)
            axes[2].set_title(f'Z-direction Rotational Acceleration ({filter_method})', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()
            
            # 调整子图间距
            plt.tight_layout()
            
            # 保存图片
            if save_plots:
                plot_filename = f'case_{case_id}_{filter_method}_waveforms.png'
                plt.savefig(os.path.join(plots_save_dir, plot_filename), dpi=300, bbox_inches='tight')
                print(f"Plot saved as {os.path.join(plots_save_dir, plot_filename)}")

            # 显示图片
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error plotting {filter_method} for case {case_id}: {str(e)}")

case_ids = [30,40,50,60]
data_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\new模型_全宽正碰结果'
if not os.path.exists(data_path):
    print(f"Error: Data path not found at {data_path}")
    exit()
save_dir = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\new模型_全宽正碰结果\filter_compare_results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# plot_filtered_comparison(case_ids, filter_methods=['CFC_filter', 'CFC_filter_ISO_butterworth', 'LX', 'CFC_filter_ISO'], save_plots=True, show_plots=False, data_path=data_path)
plot_filtered_comparison(case_ids, filter_methods=['CFC_filter_ISO_butterworth','CFC_filter_SAE', 'LX', 'CFC_filter'], save_plots=True, show_plots=False, data_path=data_path, plots_save_dir=save_dir)
# LX滤波器就是ISO中的Butterworth滤波器，只是截止频率没按照文件来；
# plot_single_filter_waveforms(case_ids, filter_method='CFC_filter_SAE', save_plots=True, show_plots=False, data_path=data_path)

# %% 根据distribution文件计算各个case在vcs中的offset量和相应墙的横纵坐标与角度
import numpy as np
import pandas as pd
from offset_cal import calculate_x_offset, calculate_y_offset

data = pd.read_csv(r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution_0917.csv")
# 使用case_id作为索引
data.set_index('case_id', inplace=True, drop=False)
# 提取的 Series 都会使用 case_id 作为索引
overlap_d = data['overlap']
angle_d = data['impact_angle']
case_ids = data['case_id'].tolist()

# 墙车几何参数
car_l_val = 2.0
car_l2_val = 0.8
car_l1_val = car_l_val - car_l2_val
wall_L_val = 4 * car_l_val
wall_W_val = car_l_val
alpha_deg_val = 50 

wall_x_origin = -0.81 # 墙角度为0时的x坐标，此时墙右侧恰与车前端接近重叠，单位为米
wall_y_origin = 0.0 # 墙角度为0时的y坐标，此时全宽正碰，单位为米

para_dict = {}

for case_id in case_ids: 

    wall_theta_deg =angle_d[case_id]  # 角度(度) 即impact_angle
    overlap = overlap_d[case_id]  # 重叠率,-1~1
    wall_L = wall_L_val # 未旋转时，墙体沿y轴方向的长度，单位为米
    wall_W = wall_W_val # 未旋转时，墙体沿x轴方向的宽度，单位为米

    car_center_cal = (wall_W_val / 2, 0)  # 车辆最前/左端中心点计算中心点，单位为米
    car_l = car_l_val # 车辆在y轴方向上的总宽度，单位为米
    car_l1 = car_l1_val # 切角后车头中间平直线段的y方向长度，单位为米
    car_alpha_deg = alpha_deg_val # 车辆前端倒角与y轴（竖直方向）的夹角，单位为度          

    para_dict.setdefault('overlap', []).append(overlap)
    para_dict.setdefault('wall_theta_deg', []).append(wall_theta_deg)

    y_offset = calculate_y_offset(
        wall_L, wall_W, wall_theta_deg, car_l, overlap
    ) # 此处单位为米
    wall_center_cal = (0, y_offset)  # 墙计算中心点，单位为米
    x_offset, _ = calculate_x_offset(
        wall_center_cal, wall_L, wall_W, wall_theta_deg,
        car_center_cal, car_l, car_l1, car_alpha_deg
    ) # 此处单位为米

    wall_center_x = np.round((wall_x_origin + x_offset)*1000, 2)  # 计算墙中心应该的的x坐标，单位为毫米
    wall_center_y =  np.round((wall_y_origin + y_offset)*1000, 3)  # 计算墙中心应该的的y坐标，单位为毫米

    para_dict.setdefault('wall_center_x_mm', []).append(wall_center_x)
    para_dict.setdefault('wall_center_y_mm', []).append(wall_center_y)
    para_dict.setdefault('x_offset_m', []).append(x_offset)
    para_dict.setdefault('y_offset_m', []).append(y_offset)

# 保存到csv文件
output_df = pd.DataFrame(para_dict, index=case_ids)
output_df.index.name = 'case_id'
output_df.to_csv(r"E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\wall_offset_from_distribution_0917.csv")
print("Wall offset parameters saved to CSV.")

# %% 计算任意给定overlap和angle的offset量和墙的横纵坐标与角度
import numpy as np
import math
from typing import Tuple

def calculate_x_offset(
    wall_center: Tuple[float, float], # 矩形墙中心点
    wall_L: float, # 未旋转时，墙体沿y轴方向的长度
    wall_W: float, # 未旋转时，墙体沿x轴方向的宽度
    wall_theta_deg: float, # 墙体围绕 wall_center 逆时针旋转的角度（单位：度）。θ > 0 为逆时针。
    car_center: Tuple[float, float], # 未切角车头线段的中心点
    car_l: float, # 车辆在y轴方向上的总宽度
    car_l1: float, # 切角后车头中间平直线段的y方向长度
    car_alpha_deg: float, # 车辆前端倒角与y轴（竖直方向）的夹角
) -> float: # 返回值：最小水平距离 (x_offset)
    """
    计算旋转的墙的右侧边与车辆前端之间的最小水平距离 (Δx)。

    Returns:
        Tuple[float, Tuple[float, float]]: 
        - 最小水平距离 (x_offset).
        - 达到最小距离时的坐标点 (x_wall, y_at_min).
    """
    # --- 输入参数数值，尤其是坐标值，是否存在无穷 ---
    if any(math.isinf(x) for x in [wall_L, wall_W, wall_theta_deg, car_l, car_l1, car_alpha_deg]) or \
       any(math.isinf(coord) for coord in wall_center + car_center):
        print("**警告: 输入参数中存在无穷大值， x_offset返回inf**")
        return float('inf'), (float('nan'), float('nan'))

    # --- 步骤 1: 计算墙的右侧边旋转后的两个端点坐标 ---
    x1, y1 = wall_center
    theta_rad = math.radians(wall_theta_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    p_top_right_local = (wall_W / 2, wall_L / 2)
    p_bottom_right_local = (wall_W / 2, -wall_L / 2)

    x_tr_rot = p_top_right_local[0] * cos_t - p_top_right_local[1] * sin_t
    y_tr_rot = p_top_right_local[0] * sin_t + p_top_right_local[1] * cos_t
    x_br_rot = p_bottom_right_local[0] * cos_t - p_bottom_right_local[1] * sin_t
    y_br_rot = p_bottom_right_local[0] * sin_t + p_bottom_right_local[1] * cos_t

    v_top_right = (x_tr_rot + x1, y_tr_rot + y1)
    v_bottom_right = (x_br_rot + x1, y_br_rot + y1)
    
    x_tr, y_tr = v_top_right
    x_br, y_br = v_bottom_right

    # --- 步骤 2: 定义车的左边界函数 x_car_left(y) ---
    x2, y2 = car_center
    x_front = x2 
    
    y_car_top_corner = y2 + car_l1 / 2
    y_car_bottom_corner = y2 - car_l1 / 2
    
    if car_alpha_deg >= 90 or car_alpha_deg <= 0:
        tan_alpha = float('inf')
    else:
        tan_alpha = math.tan(math.radians(car_alpha_deg))

    def get_x_car_left(y: float) -> float:
        if y_car_bottom_corner <= y <= y_car_top_corner:
            return x_front
        elif y > y_car_top_corner:
            return x_front + (y - y_car_top_corner) * tan_alpha
        else:
            return x_front - (y - y_car_bottom_corner) * tan_alpha

    # --- 步骤 3: 确定Y轴投影的重叠区间 ---
    y_wall_min = min(y_tr, y_br)
    y_wall_max = max(y_tr, y_br)
    y_car_min = y2 - car_l / 2
    y_car_max = y2 + car_l / 2
    
    y_overlap_start = max(y_wall_min, y_car_min)
    y_overlap_end = min(y_wall_max, y_car_max)

    if y_overlap_start >= y_overlap_end:
        print(f"**警告: Y轴重叠区间无效: {y_overlap_start} >= {y_overlap_end}**")
        print("**x_offset置为无穷大，返回无效坐标")
        return float('inf'), (float('nan'), float('nan'))

    # --- 步骤 4: 找出所有“关键Y坐标点” ---
    critical_y_coords = {y_overlap_start, y_overlap_end}
    if y_overlap_start < y_car_top_corner < y_overlap_end:
        critical_y_coords.add(y_car_top_corner)
    if y_overlap_start < y_car_bottom_corner < y_overlap_end:
        critical_y_coords.add(y_car_bottom_corner)

    # --- 步骤 5: 计算并比较，找到最终的 Δx ---
    min_gap = float('inf')
    min_gap_point_y = float('nan')

    for y_c in critical_y_coords:
        x_car = get_x_car_left(y_c)
        
        if abs(y_tr - y_br) < 1e-9:
            x_wall = max(x_tr, x_br)
        else:
            x_wall = x_br + (x_tr - x_br) * (y_c - y_br) / (y_tr - y_br)
            
        current_gap = x_car - x_wall
        
        if current_gap < min_gap:
            min_gap = current_gap
            
    x_offset = min_gap

    return x_offset

def calculate_y_offset(
    wall_L: float,
    wall_W: float,
    wall_theta_deg: float,
    car_l: float,
    overlap_y: float,
) -> float:
  wall_theta_rad = math.radians(wall_theta_deg)
  if 1 - abs(overlap_y) <= 0.01:
    return 0.0  # 基本全宽正碰时不需要偏移（前提是：车辆与墙的重叠区域足够大：wall_W/2*np.sin(wall_theta_rad) + wall_L/2*np.cos(wall_theta_rad) - car_l/2 > 0恒成立）
  elif abs(overlap_y) < 0.01:
    print(f"**警告: y方向重叠率接近0: {overlap_y}, 返回无穷大**") 
    return np.inf  # y方向重叠率接近0时，返回无穷大，表示无法计算偏移
  elif overlap_y > 0:
    y_offset = -(wall_W/2*np.sin(wall_theta_rad) + wall_L/2*np.cos(wall_theta_rad) - car_l/2) - (1 - overlap_y) * car_l
  elif overlap_y < 0:
    y_offset = (-wall_W/2*np.sin(wall_theta_rad) + wall_L/2*np.cos(wall_theta_rad) - car_l/2) + (1 + overlap_y) * car_l


  return y_offset

############################################
wall_theta_deg = 30  # 角度(度) 即impact_angle
overlap = -0.25  # 重叠率,-1~1
############################################

# 墙车几何参数
car_l_val = 2.0
car_l2_val = 0.8
car_l1_val = car_l_val - car_l2_val
wall_L_val = 4 * car_l_val
wall_W_val = car_l_val
alpha_deg_val = 50 

wall_x_origin = -0.81 # 墙角度为0时的x坐标，此时墙右侧恰与车前端接近重叠，单位为米
wall_y_origin = 0.0 # 墙角度为0时的y坐标，此时全宽正碰，单位为米
wall_L = wall_L_val # 未旋转时，墙体沿y轴方向的长度，单位为米
wall_W = wall_W_val # 未旋转时，墙体沿x轴方向的宽度，单位为米

car_center_cal = (wall_W_val / 2, 0)  # 车辆最前/左端中心点计算中心点，单位为米
car_l = car_l_val # 车辆在y轴方向上的总宽度，单位为米
car_l1 = car_l1_val # 切角后车头中间平直线段的y方向长度，单位为米
car_alpha_deg = alpha_deg_val # 车辆前端倒角与y轴（竖直方向）的夹角，单位为度          

y_offset = calculate_y_offset(
    wall_L, wall_W, wall_theta_deg, car_l, overlap
) # 此处单位为米
wall_center_cal = (0, y_offset)  # 墙计算中心点，单位为米
x_offset = calculate_x_offset(
    wall_center_cal, wall_L, wall_W, wall_theta_deg,
    car_center_cal, car_l, car_l1, car_alpha_deg
) # 此处单位为米

wall_center_x = np.round((wall_x_origin + x_offset)*1000, 2)  # 计算墙中心应该的的x坐标，单位为毫米
wall_center_y =  np.round((wall_y_origin + y_offset)*1000, 3)  # 计算墙中心应该的的y坐标，单位为毫米

# 俯视角度，左为x轴正方向，上为y轴正方向
print(f"对于 overlap={overlap}, angle={wall_theta_deg}°:")
print(f"计算得到的墙中心坐标为: ({wall_center_x} mm, {wall_center_y} mm)")
print(f"计算得到的墙偏移量为: x_offset = {x_offset:.4f} m, y_offset = {y_offset:.4f} m")

# %% 
# %% 
# %%