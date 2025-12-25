# -*- coding: utf-8 -*-
"""
MADYMO XML 自动化生成脚本 (Step 3 - New 1220 Version)
--------------------------------------------------
功能：
1. 读取 distribution.csv 参数矩阵。
2. 读取 acc_data 文件夹下的碰撞波形 (X, Y)。
3. 根据 'base_xml文件参数修改说明-副驾-50th假人-1220.pdf' 的规则，
   修改 Base XML 文件中的 DEFINE 变量和 FUNCTION.XY 表格。
4. 直接输出最终的 .xml 文件用于 MADYMO 仿真。
"""

import numpy as np
import pandas as pd
from lxml import etree
import os
import copy

# ==================== 1. 全局配置 ====================

# --- 指定本次运行要生成的 XML 类型 ---
# 格式: '{驾驶侧}_{假人体型}'
# 可选: 'DS_5th', 'DS_50th', 'DS_95th', 'PS_5th', 'PS_50th', 'PS_95th'
XML_TYPE = 'PS_50th'

# --- 路径配置 ---
# 请根据实际环境确认以下路径
BASE_XML_DIR = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\代码\MADYMO_Base_xml文件'
PARAM_FILE_PATH = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\distribution\distribution_1220.csv'
PULSE_FILES_DIR = r'G:\VCS_acc_data\acc_data_before1111_6134'
OUTPUT_DIR = r'E:\WPS Office\1628575652\WPS企业云盘\清华大学\我的企业文档\课题组相关\理想项目\仿真数据库相关\代码\output_xml_1220'

# --- Base XML 文件映射 ---
BASE_XML_PATHS = {
    'DS_5th':  os.path.join(BASE_XML_DIR, '主驾-5th假人-base-1220.xml'),
    'DS_50th': os.path.join(BASE_XML_DIR, '主驾-50th假人-base-1220.xml'),
    'DS_95th': os.path.join(BASE_XML_DIR, '主驾-95th假人-base-1220.xml'),
    'PS_5th':  os.path.join(BASE_XML_DIR, '副驾-5th假人-base-1220.xml'),
    'PS_50th': os.path.join(BASE_XML_DIR, '副驾-50th假人-base-1221.xml'),
    'PS_95th': os.path.join(BASE_XML_DIR, '副驾-95th假人-base-1220.xml'),
}

# --- 常量配置 ---
NON_DRIVER_OFFSET = 50000  # 副驾工况ID偏移量
CASE_ID_LIST = None        # 指定特定case_id列表 (None表示处理所有)
CASE_ID_LIST = [2, 50001, 50004, 50005] # 调试用

# ==================== 2. 数据映射表 & 辅助函数 ====================

# 假人 OT 映射
OT_MAPPING = {'5th': 1, '50th': 2, '95th': 3}

# D环位置 (DZ) 坐标映射
# 结构: {Is_Driver: {Level: "x y z"}}
# Is_Driver: 1=主驾, 0=副驾
DRING_COORDS_MAP = {
    1: { # MADYMO 主驾 (DS)
        4: "1.797 -0.615 0.894",          # Level 4 (Base)
        3: "1.79403 -0.62187 0.87172",    # Level 3
        2: "1.79106 -0.62874 0.84945",    # Level 2
        1: "1.78801 -0.63582 0.82651"     # Level 1
    },
    0: { # MADYMO 副驾 (PS)
        4: "1.819 0.753 0.912",           # Level 4 (Base)
        3: "1.81603 0.75987 0.88972",     # Level 3
        2: "1.81306 0.76674 0.86745",     # Level 2
        1: "1.81001 0.77382 0.84451"      # Level 1
    }
}

def parse_xml_type(xml_type):
    """解析 XML_TYPE 字符串"""
    parts = xml_type.split('_')
    side_str = parts[0]       # 'DS' or 'PS'
    percentile_str = parts[1] # '5th', '50th', '95th'
    
    is_driver = 1 if side_str == 'DS' else 0
    ot = OT_MAPPING.get(percentile_str, 2)
    return is_driver, ot, percentile_str

def calc_joint_angles(ot, Seat_X_Disp):
    """
    根据xml文件座椅位置 Seat_X_Disp 和 假人类型 OT 计算关节角度 (rad)
    """
    # 初始化
    hip, knee, ankle = 0.0, 0.0, 0.0
    
    # --- 副驾5th假人 (OT=1) ---
    if ot == 1:
        if Seat_X_Disp > 0.06: 
            hip   = 0.5 * Seat_X_Disp - 0.03     
            knee  = -3.3429 * Seat_X_Disp - 0.0309
            ankle = -7.1857 * Seat_X_Disp + 1.0216
        else: 
            hip   = 0.0
            knee  = -0.23
            ankle = 0.6

    # --- 副驾50th假人 (OT=2) ---
    elif ot == 2:
        if Seat_X_Disp > 0.005:
            hip   = -2.0703 * Seat_X_Disp + 0.1305
            knee  = 4.7459 * Seat_X_Disp - 0.6046
            ankle = -2.8728 * Seat_X_Disp + 0.1765
        else:
            hip   = 0.13
            knee  = -0.6
            ankle = 0.18

    # --- 副驾95th假人 (OT=3) ---
    elif ot == 3:
        if Seat_X_Disp > -0.006:
            hip   = -1.8182 * Seat_X_Disp + 0.0491        
            knee  = 4.2 * Seat_X_Disp - 0.38              
            ankle = -2.7273 * Seat_X_Disp + 0.0056        
        else:
            hip   = -0.05
            knee  = -0.09
            ankle = 0.6
            
    return hip, knee, ankle

def get_pulse_data_string(pulse_dir, driver_case_id, axis):
    """
    读取CSV波形并格式化为 MADYMO XY_PAIR 字符串
    """
    csv_path = os.path.join(pulse_dir, f'{axis}{driver_case_id}.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"波形文件未找到: {csv_path}")

    try:
        data = np.genfromtxt(csv_path, delimiter=None) # 自动检测分隔符, 包括逗号，制表符 \t，空格，多个连续空白字符
        
        if data.ndim == 1:
            raise ValueError("CSV解析为单列,请检查文件分隔符(应为逗号或制表符)")
        elif data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"CSV文件格式错误,应至少包含两列(时间, 加速度)。当前解析为 {data.shape}")
             
        time_ori = data[:, 0]
        acc_values = data[:, 1]
        
        # --- 1. 时间步长检测 ---
        dt_mean = np.mean(np.diff(time_ori))
        
        # 严格判断是否符合 5e-6 或 1e-5 的时间步长
        if np.isclose(dt_mean, 5e-6, atol=1e-7):
            dt = 5e-6
        elif np.isclose(dt_mean, 1e-5, atol=1e-7):
            dt = 1e-5
        else:
            raise ValueError(f"时间间隔 {dt_mean} 不符合预期的 5e-6 或 1e-5 秒。")

        # --- 2. 重新生成标准时间轴 ---
        num_points = len(acc_values)
        timestamps = np.arange(num_points) * dt
        
        # --- 3. 构建字符串 ---
        # Header 格式包含管道符 |XI YI|
        lines = ["|XI YI|"]
        
        # 循环生成数据行，根据 dt 精度动态调整格式
        for ts, acc in zip(timestamps, acc_values):
            # 保持高精度，避免科学计数法
            if dt == 5e-6:
                lines.append(f"{ts:.6f} {acc:.8f}")
            elif dt == 1e-5:
                lines.append(f"{ts:.5f} {acc:.8f}")
            else:
                lines.append(f"{ts:.7f} {acc:.8f}")
        
        # 返回拼接后的字符串 (用于插入 CDATA)
        return "\n".join(lines)

    except Exception as e:
        # 捕获 numpy 解析或其他异常，向上层抛出并附带文件名信息
        raise ValueError(f"读取或处理文件 '{csv_path}' 时出错: {e}")

# ==================== 3. XML 修改核心逻辑 ====================

def process_single_case(base_tree, case_data, case_id, is_driver, ot):
    """
    处理单个工况，返回修改后的 XML tree 对象
    """
    # 深度拷贝，防止修改影响下一次循环
    tree = copy.deepcopy(base_tree)
    root = tree.getroot()
    
    # --- 1. 参数提取与单位转换 ---
    try:
        # 获取 CSV 中的值，兼顾缩写和全称
        def get_val(keys, default=None):
            if isinstance(keys, str): keys = [keys]
            for k in keys:
                if k in case_data:
                    return case_data[k]
            # 如果是 Series，可能因为索引问题报错，这里假设列名存在
            raise KeyError(f"未找到列名: {keys}")

        val_ll1_N = get_val(['LL1']) * 1000.0          # kN -> N 
        val_ll2_N = get_val(['LL2']) * 1000.0          # kN -> N 
        val_btf_s = get_val(['BTF']) / 1000.0          # ms -> s 
        val_ptf_s = get_val(['PTF']) / 1000.0          # ms -> s 
        val_aft_s = get_val(['AFT']) / 1000.0          # ms -> s 
        val_sp_m  = get_val(['SP']) / 1000.0           # mm -> m 
        
        # 原始值读取
        raw_llattf = get_val(['LLATTF'])               # ms 
        raw_ra_deg = get_val(['RA', 'recline_angle'])  # degree 
        dz_level   = int(get_val(['DZ']))              # level 
        
    except KeyError as e:
        print(f"  [Error] Case {case_id} 数据缺失或列名错误: {e}")
        return None

    # LLATTF 逻辑 
    # 采样范围 [BTF, 150ms]。150ms 代表不切换。
    if raw_llattf > 149.99: 
        val_ll2tf_s = 1.0 # 1s 实际上代表不切换（远超仿真时长）
    else:
        val_ll2tf_s = raw_llattf / 1000.0

    # RA 逻辑 
    # VALUE = base value - deg2rad(sample - 25)
    delta_ra_rad = -np.deg2rad(raw_ra_deg - 25.0)

    # DZ 逻辑 
    val_dring_pos = DRING_COORDS_MAP[is_driver][dz_level]

    # --- 2. 修改 XML define 变量 ---

    # 单独处理 Seat_X_Disp 和 Seat_Back_rotation_Angle，因为要获取base value；且 Seat_X_Disp 影响关节角度计算
    # 处理 Seat_X_Disp = base value + SP采样值
    sp_nodes = root.xpath(".//DEFINE[@VAR_NAME='Seat_X_Disp']")
    if sp_nodes:
        base_sp_val = float(sp_nodes[0].attrib['VALUE'])
        Seat_X_Disp = base_sp_val + val_sp_m
        sp_nodes[0].attrib['VALUE'] = f"{Seat_X_Disp:.6f}"
    else:
        print(f"  [Warning] Case {case_id}: Base XML 中未找到变量 Seat_X_Disp")

    # 三个关节角度计算（基于更新后的 Seat_X_Disp）
    val_hip, val_knee, val_ankle = calc_joint_angles(ot, Seat_X_Disp)

    # Seat_Back_rotation_Angle = base value - deg2rad[RA采样值 - 25°]
    ra_nodes = root.xpath(".//DEFINE[@VAR_NAME='Seat_Back_rotation_Angle']")
    if ra_nodes:
        base_ra_val = float(ra_nodes[0].attrib['VALUE'])
        new_ra_val = base_ra_val + delta_ra_rad # 注意这里是加上 delta，因为 delta 已经是 base - sampled 的结果
        ra_nodes[0].attrib['VALUE'] = f"{new_ra_val:.6f}"
    else:
        print(f"  [Warning] Case {case_id}: Base XML 中未找到变量 Seat_Back_rotation_Angle")

    # 定义其它的要修改的变量映射 {VAR_NAME: New_Value_String}
    # 对应文档 Source 42, 49, 56, 64, 72, 78, 86, 99, 104, 109
    vars_map = {
        "R_LL1F": f"{val_ll1_N:.4f}",       # 一级限力值
        "R_LL2F": f"{val_ll2_N:.4f}",       # 二级限力值
        "RPTTF_def": f"{val_btf_s:.6f}",    # 预紧器点火
        "APTTF_def": f"{val_ptf_s:.6f}",    # 腰部预紧器点火
        "R_LL2TF": f"{val_ll2tf_s:.6f}",    # 二级限力切换
        "Dring_pos": val_dring_pos,         # D环位置
        "PAB_TTF": f"{val_aft_s:.6f}",      # 气囊点火
        "hip_angle": f"{val_hip:.6f}",      # 三个关节角度
        "knee_angle": f"{val_knee:.6f}",
        "ankle_angle": f"{val_ankle:.6f}"
    }

    # 执行 Define 替换
    for var_name, value in vars_map.items():
        # XPath 查找 <DEFINE VAR_NAME="...">
        nodes = root.xpath(f".//DEFINE[@VAR_NAME='{var_name}']")
        if nodes:
            nodes[0].attrib['VALUE'] = str(value)
        else:
            print(f"  [Warning] Case {case_id}: Base XML 中未找到变量 {var_name}")

    # --- 3. 修改碰撞波形 ---
    # 规则：直接通过 NAME 属性精确查找 FUNCTION.XY 节点
    
    # 波形文件的命名使用的是主驾 ID
    driver_case_id = case_id if is_driver else (case_id - NON_DRIVER_OFFSET)
    
    # 修改配置：移除 ID 依赖，使用精确的 NAME 
    pulse_configs = [
        {'axis': 'x', 'exact_name': 'X_lin_pulse_fun'},
        {'axis': 'y', 'exact_name': 'Y_lin_pulse_fun'}
    ]

    for p_conf in pulse_configs:
        try:
            # 读取并格式化波形数据
            pulse_str = get_pulse_data_string(PULSE_FILES_DIR, driver_case_id, p_conf['axis'])
            
            # 定位节点: <FUNCTION.XY NAME="...">
            # 使用 local-name() 处理可能的命名空间问题，并精确匹配 NAME 属性
            target_name = p_conf['exact_name']
            # XPath 解释：查找任意层级下，标签名为 FUNCTION.XY 且 NAME 属性等于目标值的节点
            func_nodes = root.xpath(f".//*[local-name()='FUNCTION.XY'][@NAME='{target_name}']")
            
            if func_nodes:
                func_node = func_nodes[0]
                # 找到子节点 TABLE
                table_node = func_node.find("TABLE")
                if table_node is not None:
                    # 使用 CDATA 包装数据
                    table_node.text = etree.CDATA(pulse_str)
                else:
                    print(f"  [Error] Case {case_id}: 找到 FUNCTION.XY (NAME={target_name}) 但未找到 TABLE 节点")
            else:
                print(f"  [Error] Case {case_id}: 未找到 NAME='{target_name}' 的 FUNCTION.XY 节点")

        except Exception as e:
            print(f"  [Error] Case {case_id}: 处理波形 {p_conf['axis']} 失败 - {str(e)}")
            return None

    return tree

# ==================== 4. 主程序 ====================

def generate_xml_files():
    print("="*60)
    print(f"开始生成 MADYMO XML 文件 - 类型: {XML_TYPE}")
    print("="*60)

    # 1. 解析配置
    is_driver_side, ot, percentile_str = parse_xml_type(XML_TYPE)
    base_xml_path = BASE_XML_PATHS.get(XML_TYPE)
    
    if not os.path.exists(base_xml_path):
        print(f"[Fatal] Base XML 文件不存在: {base_xml_path}")
        return

    # 2. 读取参数矩阵
    print(f"正在读取参数文件: {PARAM_FILE_PATH}")
    if PARAM_FILE_PATH.endswith('.csv'):
        df = pd.read_csv(PARAM_FILE_PATH)
    else:
        data = np.load(PARAM_FILE_PATH, allow_pickle=True)
        df = pd.DataFrame({k: data[k] for k in data.files})
        
    # 设 case_id 为索引方便查找
    if 'case_id' in df.columns:
        df.set_index('case_id', inplace=True)
    
    # 3. 筛选要处理的 Case
    # 条件：
    # (1) 对应侧 (主驾或副驾)
    # (2) 对应假人类型 (OT)
    # (3) Pulse OK
    # (4) Injury Not OK (只跑还没跑过的或失败的)
    
    # 获取波形目录下的所有主驾 ID
    valid_driver_ids = []
    if os.path.exists(PULSE_FILES_DIR):
        for f in os.listdir(PULSE_FILES_DIR):
            if f.startswith('x') and f.endswith('.csv'):
                try:
                    valid_driver_ids.append(int(f[1:-4]))
                except:
                    pass
    valid_driver_ids = set(valid_driver_ids)

    # 构造待处理列表
    tasks = []
    for cid in df.index:
        # 如果指定了 list，则只处理 list 中的
        if CASE_ID_LIST is not None and cid not in CASE_ID_LIST:
            continue

        row = df.loc[cid]
        
        # 检查是否为主副驾
        # 优先读取 'is_driver_side'，如果没有则根据 ID 判断: 主驾 <=50000, 副驾 >50000
        current_is_driver = row.get('is_driver_side', cid <= 50000)
        
        # 匹配配置的驾驶侧
        if current_is_driver != is_driver_side:
            continue
            
        # 匹配假人 OT
        row_ot = row.get('OT', row.get('occupant_type'))
        if row_ot != ot:
            continue
            
        # 检查波形是否可用的标志位, 必须为True
        if row.get('is_pulse_ok') != True:
            continue
            
        # 检查波形文件是否存在
        driver_id = cid if is_driver_side else (cid - NON_DRIVER_OFFSET)
        if driver_id not in valid_driver_ids:
            continue

        ###################################################################
        # is_injury_ok为true或false都不处理
        if row.get('is_injury_ok') == True or row.get('is_injury_ok') == False:
            continue
        ###################################################################

        tasks.append(cid)

    print(f"筛选出 {len(tasks)} 个待处理工况。")
    
    # 4. 加载 Base XML
    # parser 使用 remove_blank_text=False 保留格式，但 lxml 默认会在写入时重排
    parser = etree.XMLParser(remove_blank_text=False, strip_cdata=False, huge_tree=True)
    base_tree = etree.parse(base_xml_path, parser)

    # 5. 循环生成
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    success_count = 0
    
    for case_id in tasks:
        try:
            # 处理单个工况
            print(f"Processing Case ID: {case_id} ...")
            new_tree = process_single_case(base_tree, df.loc[case_id], case_id, is_driver_side, ot)
            
            if new_tree:
                out_name = f"madymo_{case_id}.xml"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                
                # 写入文件
                # encoding='UTF-8', xml_declaration=True
                new_tree.write(out_path, encoding='UTF-8', xml_declaration=True, pretty_print=True)
                
                success_count += 1
                if success_count % 50 == 0:
                    print(f"已生成 {success_count} 个文件...")
        except Exception as e:
            print(f"Case {case_id} 处理发生未知异常: {e}")

    print("="*60)
    print(f"处理完成！")
    print(f"成功: {success_count} / 总任务: {len(tasks)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)

if __name__ == '__main__':
    generate_xml_files()