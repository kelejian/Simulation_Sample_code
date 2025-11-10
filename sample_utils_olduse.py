# %% 为ditribution文件添加几个手动指定的case
import numpy as np
import pandas as pd
distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1109.csv'
new_distribution_path = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1110.csv'
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
# 手动指定的case列表及对应参数。其余参数均为NaN
# 重叠率列表,从0.45到0.65，间隔0.01；caseid从8563开始
overlap_lis1 = np.arange(0.45, 0.66, 0.01).tolist()
caseid_lis1 = [8563 + i for i in range(len(overlap_lis1))]
velocity_lis1 = [36.0] * len(overlap_lis1)
angle_lis1 = [-30.0] * len(overlap_lis1)
is_driver_side_lis1 = [1] * len(overlap_lis1)
have_run_lis1 = [False] * len(overlap_lis1)

# 重叠率列表，从-0.85到-0.64，间隔0.01
overlap_lis2 = np.arange(-0.85, -0.64, 0.01).tolist()
caseid_lis2 = [caseid_lis1[-1]+1 + i for i in range(len(overlap_lis2))]
velocity_lis2 = [42.0] * len(overlap_lis2)
angle_lis2 = [30.0] * len(overlap_lis2)
is_driver_side_lis2 = [1] * len(overlap_lis2)
have_run_lis2 = [False] * len(overlap_lis2)

# 增加若干45°，变化重叠率的case
overlap_lis3 = np.arange(-0.6, -0.39, 0.01).tolist() # 从-0.6到-0.4
caseid_lis3 = [caseid_lis2[-1]+1 + i for i in range(len(overlap_lis3))]
velocity_lis3 = [32.0] * len(overlap_lis3)
angle_lis3 = [45.0] * len(overlap_lis3)
is_driver_side_lis3 = [1] * len(overlap_lis3)
have_run_lis3 = [False] * len(overlap_lis3)

# 增加若干-45°，变化重叠率的case
overlap_lis4 = np.arange(0.4, 0.61, 0.01).tolist() # 从0.4到0.6
caseid_lis4 = [caseid_lis3[-1]+1 + i for i in range(len(overlap_lis4))]
velocity_lis4 = [32.0] * len(overlap_lis4)
angle_lis4 = [-45.0] * len(overlap_lis4)
is_driver_side_lis4 = [1] * len(overlap_lis4)
have_run_lis4 = [False] * len(overlap_lis4)

# 合并所有列表为一个lis
overlap_lis = overlap_lis1 + overlap_lis2 + overlap_lis3 + overlap_lis4
caseid_lis = caseid_lis1 + caseid_lis2 + caseid_lis3 + caseid_lis4
velocity_lis = velocity_lis1 + velocity_lis2 + velocity_lis3 + velocity_lis4
angle_lis = angle_lis1 + angle_lis2 + angle_lis3 + angle_lis4
is_driver_side_lis = is_driver_side_lis1 + is_driver_side_lis2 + is_driver_side_lis3 + is_driver_side_lis4
have_run_lis = have_run_lis1 + have_run_lis2 + have_run_lis3 + have_run_lis4

params_add_driver_side = {
    'case_id': caseid_lis,
    'impact_velocity': velocity_lis,
    'impact_angle': angle_lis,
    'overlap': overlap_lis,
    'is_driver_side': is_driver_side_lis,
    'have_run': have_run_lis,
}
is_driver_side_lis_no = [0]*len(caseid_lis)
caseid_lis_no = [caseid + 50000 for caseid in caseid_lis]
params_add_driver_side_no = {
    'case_id': caseid_lis_no,
    'impact_velocity': velocity_lis,
    'impact_angle': angle_lis,
    'overlap': overlap_lis,
    'is_driver_side': is_driver_side_lis_no,
    'have_run': have_run_lis,
}
# 合并两个字典为:params_add
params_add = {}
for key in params_add_driver_side.keys():
    params_add[key] = params_add_driver_side[key] + params_add_driver_side_no[key]

# 创建DataFrame
df_add = pd.DataFrame(params_add).set_index('case_id', drop=False)
# 将手动指定的参数添加到distribution_df中。如果已有对应case_id，如果have_run为True则跳过，否则直接替换掉
for case_id, row in df_add.iterrows():
    if case_id in distribution_df.index:
        if distribution_df.at[case_id, 'have_run']:
            print(f"Case {case_id} already exists and have_run is True, skipping...")
            continue
    distribution_df.loc[case_id] = row
    print(f"Added/Updated case {case_id} to distribution DataFrame.")

print(f"Added {len(df_add)} manual cases to distribution DataFrame.")


# 保存更新后的distribution文件
if new_distribution_path.endswith('.npz'):
    np.savez(new_distribution_path, **{col: distribution_df[col].values for col in distribution_df.columns})
elif new_distribution_path.endswith('.csv'):
    distribution_df.to_csv(new_distribution_path, index=False)
    print("Updated distribution file has been saved.")

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
file1 = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1106.csv'
file2 = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1108.csv'

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
            print(f"第 {i+1} 行case{row1['case_id']}不同: {diff_details}")
    
    if not differences:
        print("两个文件内容完全一致。")
    else:
        print(f"总共有 {len(differences)} 行内容不同。")


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

# %% 为distribution文件添加一列active_learning，初始值均为NaN
import numpy as np
import pandas as pd
distribution_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1006_V4.csv'
new_distribution_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1006_V5.csv'
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
# 如果distribution_df中没有injury_active_learning_generated列，先在DataFrame中添加该列，初始值为NaN
if 'injury_active_learning_generated' not in distribution_df.columns:
    distribution_df['injury_active_learning_generated'] = np.nan
    print("Added missing column 'injury_active_learning_generated' to distribution DataFrame.")
if 'injury_active_learning_used' not in distribution_df.columns:
    distribution_df['injury_active_learning_used'] = np.nan
    print("Added missing column 'injury_active_learning_used' to distribution DataFrame.")
# 保存更新后的distribution文件
if new_distribution_path.endswith('.npz'):
    np.savez(new_distribution_path, **{col: distribution_df[col].values for col in distribution_df.columns})
elif new_distribution_path.endswith('.csv'):
    distribution_df.to_csv(new_distribution_path, index=False)
    print("Updated distribution file with injury_active_learning columns has been saved.")

# %% 把distribution文件中aav_status列中为0的case，其ttf列值改为0; lla_status列中为0的case，其llattf列值改为0
import numpy as np
import pandas as pd
distribution_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1009.csv'
new_distribution_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1010.csv'
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
# aav_status列中为0的case，其ttf列值改为0.0
mask_aav = distribution_df['aav_status'] == 0
distribution_df.loc[mask_aav, 'ttf'] = 0.0
print(f"Set ttf=0 for {mask_aav.sum()} cases where aav_status=0.")

# lla_status列中为0的case，其llattf列值改为0.0
mask_lla = distribution_df['lla_status'] == 0
distribution_df.loc[mask_lla, 'llattf'] = 0.0
print(f"Set llattf=0 for {mask_lla.sum()} cases where lla_status=0.")
# 保存更新后的distribution文件
if new_distribution_path.endswith('.npz'):
    np.savez(new_distribution_path, **{col: distribution_df[col].values for col in distribution_df.columns})
elif new_distribution_path.endswith('.csv'):
    distribution_df.to_csv(new_distribution_path, index=False)
    print("Updated distribution file with ttf and llattf corrections has been saved.")

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

# %% 在case_id列后面插入一列is_driver_side, 表示是否为主驾侧case，主驾侧为1，副驾侧为0；初始化为1
import pandas as pd
distribution_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1016.csv'
new_distribution_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1017.csv'
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
# 在case_id列后面插入一列is_driver_side, 初始化为1
distribution_df.insert(1, 'is_driver_side', 1)
# 保存更新后的distribution文件
if new_distribution_path.endswith('.npz'):
    np.savez(new_distribution_path, **{col: distribution_df[col].values for col in distribution_df.columns})
elif new_distribution_path.endswith('.csv'):
    distribution_df.to_csv(new_distribution_path, index=False)
    print("Updated distribution file with is_driver_side has been saved.")


# %% 为当前distribution文件初始化副驾侧的行信息
# 在文件末尾空一行，然后添加副驾侧的行，内容为：将前面主驾侧的行中的case_id复制过来（并加50000）、并把对应的have_run, is_pulse_ok, impact_velocity	impact_angle	overlap, delta_vx(kph)	delta_vy(kph)	delta_v(kph) 的值都复制过来，并把is_driver_side设置为0，其它保持nan
import numpy as np
import pandas as pd
distribution_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1017.csv'
new_distribution_path = r'E:\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1017_V2.csv'
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

# 需要复制的列
columns_to_copy = ['have_run', 'is_pulse_ok', 'impact_velocity', 'impact_angle', 'overlap', 'delta_vx(kph)', 'delta_vy(kph)', 'delta_v(kph)']
# 遍历当前DataFrame，找到is_driver_side==1的行，复制并修改后添加到DataFrame末尾，除了需要复制的列，其它列保持nan
new_rows = []
for idx, row in distribution_df.iterrows():
    if row['is_driver_side'] == 1:
        new_row = {col: np.nan for col in distribution_df.columns}
        new_row['case_id'] = row['case_id'] + 50000
        new_row['is_driver_side'] = 0
        for col in columns_to_copy:
            new_row[col] = row[col]
        new_rows.append(new_row)

# 将新行添加到DataFrame末尾
if new_rows:
    distribution_df = pd.concat([distribution_df, pd.DataFrame(new_rows)], ignore_index=True)
    print(f"Added {len(new_rows)} passenger side cases to distribution DataFrame.")
else:
    print("No driver side cases found to copy.")

# 保存更新后的distribution文件
if new_distribution_path.endswith('.npz'):
    np.savez(new_distribution_path, **{col: distribution_df[col].values for col in distribution_df.columns})
elif new_distribution_path.endswith('.csv'):
    distribution_df.to_csv(new_distribution_path, index=False)
    print("Updated distribution file with passenger side cases has been saved.")



# %% 