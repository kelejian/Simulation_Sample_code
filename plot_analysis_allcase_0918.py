import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

# --- 1. 核心绘图与分析函数 ---

def plot_waveforms_on_single_figure(loaded_data, condition_str, save_dir):
    """将多个case的波形绘制在同一张图的三个子图上。"""
    if not loaded_data:
        print("No data to plot.")
        return "no_plot.png"

    print(f"Plotting aggregated waveforms for condition: {condition_str}...")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))
    title = f'Aggregated Waveforms for Condition: {condition_str}\n({len(loaded_data)} cases)'
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for case_id, data_dict in loaded_data.items():
        axes[0].plot(data_dict['x']['time'] * 1000, data_dict['x']['ax'], color='blue', alpha=0.3, linewidth=1)
        axes[1].plot(data_dict['y']['time'] * 1000, data_dict['y']['ay'], color='red', alpha=0.3, linewidth=1)
        axes[2].plot(data_dict['z']['time'] * 1000, data_dict['z']['az'], color='green', alpha=0.3, linewidth=1)

    axes[0].set_title('X-direction Acceleration', fontsize=12)
    axes[0].set_ylabel('Acceleration (m/s²)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title('Y-direction Acceleration', fontsize=12)
    axes[1].set_ylabel('Acceleration (m/s²)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[2].set_title('Z-direction Rotational Acceleration', fontsize=12)
    axes[2].set_ylabel('Angular Acceleration (rad/s²)', fontsize=12)
    axes[2].set_xlabel('Time (ms)', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    # # --------------设置子图y轴显示范围--------------
    # axes[0].set_ylim(-650, 250)  # X方向加速度范围
    # axes[1].set_ylim(-300, 300)  # Y方向加速度
    # axes[2].set_ylim(-300, 300)  # Z方向旋转加速度
    # # ---------------------------------------------

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    safe_filename = condition_str.replace(' ', '_').replace('<=', 'le').replace('>=', 'ge').replace('<', 'lt').replace('>', 'gt').replace('==', 'eq')
    plot_filename = f'waveforms_{safe_filename}.png'
    save_path = os.path.join(save_dir, plot_filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Aggregated waveform plot saved to {save_path}")
    return plot_filename


def plot_statistical_distributions(peak_values_dict, condition_str, save_dir):
    """
    绘制峰值分布的直方图 (3x3布局)。
    行: X, Y, Z方向
    列: 正向峰值, 负向峰值, 绝对值峰值
    """
    print("Plotting detailed statistical distributions...")
    # 使用更大的figsize以容纳3x3的子图
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    title = f'Peak Acceleration Distribution\nCondition: {condition_str}'
    fig.suptitle(title, fontsize=18, fontweight='bold')

    directions = ['x', 'y', 'z']
    peak_types = ['pos', 'neg', 'abs']
    
    # 设置列标题
    col_titles = ['Positive Peaks', 'Negative Peaks', 'Absolute Peaks']
    for ax, col_title in zip(axes[0], col_titles):
        ax.set_title(col_title, fontsize=14, fontweight='bold')

    # 设置行标签
    row_labels = ['X-direction (m/s²)', 'Y-direction (m/s²)', 'Z-direction (rad/s²)']
    for ax, row_label in zip(axes[:,0], row_labels):
        ax.set_ylabel(f'Frequency\n({row_label})', fontsize=12)

    for i, direction in enumerate(directions):
        for j, peak_type in enumerate(peak_types):
            peaks = peak_values_dict[direction][peak_type]
            ax = axes[i, j]
            if not peaks: continue
            
            sns.histplot(peaks, kde=True, ax=ax, bins=30)
            mean_val = np.mean(peaks)
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.set_xlabel('Peak Acceleration', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    safe_filename = condition_str.replace(' ', '_').replace('<=', 'le').replace('>=', 'ge').replace('<', 'lt').replace('>', 'gt').replace('==', 'eq')
    plot_filename = f'distribution_detailed_{safe_filename}.png'
    save_path = os.path.join(save_dir, plot_filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Detailed distribution plot saved to {save_path}")
    return plot_filename


# --- 2. 模块化的统计分析子函数  ---

def analyze_peak_values(loaded_data):
    """计算正向、负向和绝对值峰值的基本统计信息。"""
    # 更精细的数据结构
    peak_values = {
        'x': {'pos': [], 'neg': [], 'abs': []},
        'y': {'pos': [], 'neg': [], 'abs': []},
        'z': {'pos': [], 'neg': [], 'abs': []}
    }
    
    for case_id, data_dict in loaded_data.items():
        peak_values['x']['pos'].append(data_dict['x']['ax'].max())
        peak_values['x']['neg'].append(data_dict['x']['ax'].min())
        peak_values['x']['abs'].append(data_dict['x']['ax'].abs().max())
        
        peak_values['y']['pos'].append(data_dict['y']['ay'].max())
        peak_values['y']['neg'].append(data_dict['y']['ay'].min())
        peak_values['y']['abs'].append(data_dict['y']['ay'].abs().max())

        peak_values['z']['pos'].append(data_dict['z']['az'].max())
        peak_values['z']['neg'].append(data_dict['z']['az'].min())
        peak_values['z']['abs'].append(data_dict['z']['az'].abs().max())
    
    stats = {}
    for direction, p_types in peak_values.items():
        stats[direction] = {}
        for p_type, peaks in p_types.items():
            if peaks:
                stats[direction][p_type] = {
                    'mean': np.mean(peaks), 'std': np.std(peaks),
                    'min': np.min(peaks), 'max': np.max(peaks),
                    'median': np.median(peaks)
                }
            else:
                stats[direction][p_type] = {k: 0 for k in ['mean', 'std', 'min', 'max', 'median']}
            
    return stats, peak_values


def analyze_threshold_exceedance(peak_values, thresholds):
    """根据传入的峰值数据和精细阈值，计算超过阈值的case比例。"""
    results = {}
    total_cases = len(peak_values['x']['pos']) # Use any list to get total count
    if total_cases == 0:
        return {}

    for direction, th_values in thresholds.items():
        results[direction] = {}
        # 正向阈值分析
        if 'pos' in th_values:
            count = sum(1 for p in peak_values[direction]['pos'] if p > th_values['pos'])
            results[direction]['pos'] = {'count': count, 'percentage': (count / total_cases) * 100}
        # 负向阈值分析
        if 'neg' in th_values:
            count = sum(1 for p in peak_values[direction]['neg'] if p < th_values['neg'])
            results[direction]['neg'] = {'count': count, 'percentage': (count / total_cases) * 100}
            
    return results

def save_peak_stats_to_csv(peak_stats, file_path):
    """
    【新增】将峰值统计数据保存为CSV文件。
    """
    data_for_df = []
    for direction, p_types in peak_stats.items():
        for p_type, stats in p_types.items():
            row = {
                'Direction': direction.upper(),
                'Peak Type': p_type.capitalize(),
                'Mean': stats['mean'],
                'Std Dev': stats['std'],
                'Median': stats['median'],
                'Min': stats['min'],
                'Max': stats['max']
            }
            data_for_df.append(row)
    
    df = pd.DataFrame(data_for_df)
    df.to_csv(file_path, index=False, float_format='%.2f')
    print(f"Peak statistics saved to {file_path}")


def save_threshold_results_to_csv(threshold_results, thresholds, file_path):
    """
    【新增】将阈值分析结果保存为CSV文件。
    """
    data_for_df = []
    for direction, results in threshold_results.items():
        for p_type, result_data in results.items():
            condition_value = thresholds[direction][p_type]
            condition_str = f"> {condition_value}" if p_type == 'pos' else f"< {condition_value}"
            row = {
                'Direction': direction.upper(),
                'Condition': condition_str,
                'Cases Exceeding': result_data['count'],
                'Percentage': result_data['percentage']
            }
            data_for_df.append(row)
            
    df = pd.DataFrame(data_for_df)
    df.to_csv(file_path, index=False, float_format='%.2f')
    print(f"Threshold analysis results saved to {file_path}")

# --- 3. 主流程控制函数 ---

def _generate_query_part(param_name, range_spec):
    """
    【新增辅助函数】根据范围定义生成pandas查询子字符串。
    
    :param param_name: DataFrame中的列名 (e.g., 'impact_velocity')
    :param range_spec: 范围定义。可以是 None, [min, max], 或 [[min1, max1], [min2, max2]]
    :return: pandas查询字符串的一部分
    """
    if range_spec is None:
        return ""
    
    # 检查是否为多个范围 (列表的列表)
    if isinstance(range_spec[0], list):
        sub_queries = []
        for sub_range in range_spec:
            sub_queries.append(f"({sub_range[0]} <= {param_name} <= {sub_range[1]})")
        # 用 'or' 连接多个范围条件
        return f"({' or '.join(sub_queries)})"
    else:
        # 单个范围
        return f"({range_spec[0]} <= {param_name} <= {range_spec[1]})"

def analyze_and_plot_cases_by_condition(params_df, data_dir, output_root_dir, vel_range, ang_range, ov_range, thresholds):
    """
    增加调用CSV保存函数的步骤。
    """
    # ... (前面的代码部分，从构建查询字符串到加载数据，都保持不变) ...
    # 1. 构建查询字符串
    query_parts = []
    vel_query = _generate_query_part('impact_velocity', vel_range)
    if vel_query: query_parts.append(vel_query)
    ang_query = _generate_query_part('impact_angle', ang_range)
    if ang_query: query_parts.append(ang_query)
    ov_query = _generate_query_part('overlap', ov_range)
    if ov_query: query_parts.append(ov_query)
    query_str = " & ".join(query_parts)
    
    if not query_str:
        print("Error: No filtering conditions provided."); return
        
    filtered_df = params_df.query(query_str)
    # ********************************************************************************************************************************************
    # ********************************************************************************************************************************************

    # 只处理have_run为True的case
    filtered_df = filtered_df[filtered_df['have_run'] == True]
    # ********************************************************************************************************************************************
    # ********************************************************************************************************************************************
    case_ids_to_process = filtered_df.index.tolist()
    
    if not case_ids_to_process:
        print(f"No cases found for the condition: {query_str}"); return

    print(f"Found {len(case_ids_to_process)} cases for condition: {query_str}")

    # 2. 创建输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    condition_name = query_str.replace(' ', '').replace('<=', 'le').replace('>=', 'ge').replace('<', 'lt').replace('>', 'gt').replace('&', '_').replace('|','or')
    analysis_dir = os.path.join(output_root_dir, f"analysis_{condition_name}_{timestamp}")
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"Output will be saved to: {analysis_dir}")

    # 3. 加载数据
    loaded_data = {}
    for case_id in case_ids_to_process:
        files = {'x': os.path.join(data_dir, f'x{case_id}.csv'), 'y': os.path.join(data_dir, f'y{case_id}.csv'), 'z': os.path.join(data_dir, f'z{case_id}.csv')}
        if all(os.path.exists(f) for f in files.values()):
            loaded_data[case_id] = {
                'x': pd.read_csv(files['x'], sep='\t', header=None, names=['time', 'ax']),
                'y': pd.read_csv(files['y'], sep='\t', header=None, names=['time', 'ay']),
                'z': pd.read_csv(files['z'], sep='\t', header=None, names=['time', 'az'])
            }
        # else:
        #     print(f"Warning: Data for case {case_id} is incomplete. Skipping.")

    # 4. 执行绘图
    agg_plot_filename = plot_waveforms_on_single_figure(loaded_data, query_str, analysis_dir)

    # 5. 执行统计分析
    peak_stats, peak_values = analyze_peak_values(loaded_data)
    threshold_results = analyze_threshold_exceedance(peak_values, thresholds)
    dist_plot_filename = plot_statistical_distributions(peak_values, query_str, analysis_dir)

    # 6. 【新增】保存统计结果到CSV文件
    peak_stats_path = os.path.join(analysis_dir, 'peak_statistics.csv')
    save_peak_stats_to_csv(peak_stats, peak_stats_path)
    
    threshold_results_path = os.path.join(analysis_dir, 'threshold_analysis.csv')
    save_threshold_results_to_csv(threshold_results, thresholds, threshold_results_path)

    # 7. 生成Markdown报告 (原步骤6)
    report_path = os.path.join(analysis_dir, 'statistical_report_detailed.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        # ... (Markdown报告的生成逻辑完全不变) ...
        f.write(f"# Detailed Analysis Report for Condition: `{query_str}`\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Cases Found:** {len(case_ids_to_process)}\n")
        f.write(f"**Total Cases Processed (with complete data):** {len(loaded_data)}\n\n")
        
        f.write("## 1. Peak Acceleration Statistics\n\n")
        f.write(f"Data saved to `peak_statistics.csv`\n\n") # 可选：在报告中提示
        f.write("| Direction | Peak Type | Mean | Std Dev | Median | Min | Max |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for d in ['x', 'y', 'z']:
            for p_type in ['pos', 'neg', 'abs']:
                stats = peak_stats[d][p_type]
                f.write(f"| **{d.upper()}** | {p_type.capitalize()} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['median']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |\n")
        
        f.write("\n## 2. Threshold Exceedance Analysis\n\n")
        f.write(f"Data saved to `threshold_analysis.csv`\n\n") # 可选：在报告中提示
        f.write("| Direction | Condition | Cases Exceeding | Percentage |\n")
        f.write("|---|---|---|---|\n")
        for d in ['x', 'y', 'z']:
            if d in thresholds and 'pos' in thresholds[d]:
                res = threshold_results[d]['pos']
                f.write(f"| **{d.upper()}** | > {thresholds[d]['pos']} | {res['count']} | {res['percentage']:.2f}% |\n")
            if d in thresholds and 'neg' in thresholds[d]:
                res = threshold_results[d]['neg']
                f.write(f"| **{d.upper()}** | < {thresholds[d]['neg']} | {res['count']} | {res['percentage']:.2f}% |\n")
            
        f.write("\n## 3. Visualizations\n\n")
        f.write("### Aggregated Waveforms\n\n")
        f.write(f"![Aggregated Waveforms]({agg_plot_filename})\n\n")
        f.write("### Detailed Peak Value Distributions\n\n")
        f.write(f"![Peak Distributions]({dist_plot_filename})\n\n")

    print(f"Detailed statistical report saved to {report_path}")

# --- 4. 主执行块 ---
if __name__ == "__main__":
    # --- 配置区域 ---
    # 1. 设置路径
    data_directory = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\acc_data_before0915'
    case_params_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution_0917.csv'
    output_root_directory = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\acc_data_before0915\analysis_reports'

    # 2. 定义您想分析的工况范围
    velocity_range = [23, 65]
    angle_range = None
    overlap_range = None

    # 3. 定义想分析的阈值
    # 格式: {'direction': {'pos': positive_threshold, 'neg': negative_threshold}}
    # 如果某个方向或类型不分析，可以省略对应的键。
    analysis_thresholds = {
        'x': {'pos': 150, 'neg': -600}, 
        'y': {'pos': 200, 'neg': -200}, 
        'z': {'pos': 200, 'neg': -200} 
    }

    # --- 执行区域 ---
    os.makedirs(output_root_directory, exist_ok=True)
    print("Loading case parameters...")
    try:
        if case_params_path.endswith('.npz'):
            all_case_params = np.load(case_params_path, allow_pickle=True)
        elif case_params_path.endswith('.csv'):
            all_case_params = pd.read_csv(case_params_path)
        # **********************************************************************
        params_df = pd.DataFrame({
        'case_id': all_case_params['case_id'],
        'have_run': all_case_params['have_run'],
        'is_pulse_ok': all_case_params['is_pulse_ok'],
        'impact_velocity': all_case_params['impact_velocity'],
        'impact_angle': all_case_params['impact_angle'],
        'overlap': all_case_params['overlap']
        }).set_index('case_id')
        # **********************************************************************
        print("Parameters loaded successfully.")
        
    except Exception as e:
        print(f"Error loading case parameters file at {case_params_path}: {e}")
        exit()

    # 在函数调用时传入 analysis_thresholds
    analyze_and_plot_cases_by_condition(
        params_df=params_df,
        data_dir=data_directory,
        output_root_dir=output_root_directory,
        vel_range=velocity_range,
        ang_range=angle_range,
        ov_range=overlap_range,
        thresholds=analysis_thresholds
    )