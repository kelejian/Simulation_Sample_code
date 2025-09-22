# -*- coding: utf-8 -*-
"""
本代码根据仿真参数矩阵(distribution.npz)和碰撞波形文件(CSV)，
为每个工况自动化生成MADYMO仿真所需的.var输入文件。
"""

import numpy as np
from lxml import etree
import os
import pandas as pd

# --- 1. 全局配置 ---

# distribution参数文件路径
PARAM_FILE_PATH = r'I:\000 LX\dataset0715\02\distribution_0923_V2.csv'
# .var模板文件路径
BASE_VAR_FILE_PATH = 'basic_variables.var'
# 碰撞波形CSV文件所在的目录
PULSE_FILES_DIR = r'I:\000 LX\dataset0715\03\acceleration_data_200ms'
# 生成的.var文件保存的目录
OUTPUT_DIR = './output_vars/'

# --- 代码主体 ---

def generate_var_files():
    """
    主函数，执行所有文件的生成任务。
    """
    # --- 2. 加载参数矩阵 ---
    try:
        if PARAM_FILE_PATH.endswith('.npz'):
            data = np.load(PARAM_FILE_PATH, allow_pickle=True)
            data = pd.DataFrame({
                key: data[key]
                for key in data.files
            }).set_index('case_id')
        elif PARAM_FILE_PATH.endswith('.csv'):
            data = pd.read_csv(PARAM_FILE_PATH)
            data.set_index('case_id', inplace=True)
        else:
            raise ValueError("参数文件必须是.npz或.csv格式。")
        print(f"成功加载 '{PARAM_FILE_PATH}' 文件。")
    except FileNotFoundError:
        print(f"错误：无法找到参数文件 '{PARAM_FILE_PATH}'。请检查文件路径。")
        return

    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: '{OUTPUT_DIR}'")

    # --- 3. 核心处理循环 ---
    # 从PULSE_FILES_DIR目录下读取以x开头，.csv结尾的文件，统计文件数量作为样本数量, 并得到caseid列表
    case_ids = []
    # case_paths = []
    if not os.path.exists(PULSE_FILES_DIR):
        print(f"目录 {PULSE_FILES_DIR} 不存在，请检查路径。")
    for root, dirs, files in os.walk(PULSE_FILES_DIR):
        for file in files:
            if file.startswith('x') and file.endswith('.csv'):
                case_id = int(file.split('.')[0][1:])
                case_ids.append(case_id)
                # case_paths.append(os.path.join(root, file))
    case_ids.sort()
    # case_paths.sort(key=lambda x: int(x.split(os.sep)[-1].split('.')[0][1:]))
    print(f"在{PULSE_FILES_DIR}找到 {len(case_ids)} 个x csv文件。")

    for case_id in case_ids:
        
        # ----------------------------------------------
        # 决定哪些工况跳过
        # if case_id > 100:
        #     break
        # ----------------------------------------------
        print(f"正在处理工况: {case_id}...")
        # --- 3.1. 获取当前工况的参数值 ---
        
        if case_id not in data.index:
            print(f"  - !!警告：工况ID {case_id} 在参数文件中未找到，跳过该工况。")
            continue
        # 跳过碰撞波形数据有问题的工况(is_pulse_ok=False)
        if data.loc[case_id, 'is_pulse_ok'] == False:
            # print(f"  - !!警告：工况ID {case_id} 的碰撞波形数据有问题，跳过该工况。")
            continue

        # 注意data是一个DataFrame，可以直接通过loc按行索引取值
        occupant_type = data.loc[case_id, 'occupant_type']
        ll1_kn = data.loc[case_id, 'll1']
        ll2_kn = data.loc[case_id, 'll2']
        btf_ms = data.loc[case_id, 'btf']
        pp_mm = data.loc[case_id, 'pp']
        ptf_ms = data.loc[case_id, 'ptf']
        plp_mm = data.loc[case_id, 'plp']
        lla_status = data.loc[case_id, 'lla_status']
        llattf_ms = data.loc[case_id, 'llattf']
        dz_level = data.loc[case_id, 'dz']
        aft_ms = data.loc[case_id, 'aft']
        aav_status = data.loc[case_id, 'aav_status']
        ttf_ms = data.loc[case_id, 'ttf']
        sp_mm = data.loc[case_id, 'sp']
        recline_deg = data.loc[case_id, 'recline_angle']

        # 如果上述参数中有任何NaN值或者缺失值，则跳过该工况
        param_values = [occupant_type, ll1_kn, ll2_kn, btf_ms, pp_mm, ptf_ms, plp_mm,
                       lla_status, llattf_ms, dz_level, aft_ms, aav_status, ttf_ms,
                       sp_mm, recline_deg]
        
        # 检查NaN值、None值和空字符串
        if (pd.isnull(param_values).any() or 
            any(val == '' for val in param_values if isinstance(val, str))):
            print(f"  - !!警告：工况ID {case_id} 存在缺失参数，跳过该工况。")
            continue
   

        # --- 3.2. 解析.var模板文件 ---
        try:
            var_tree = etree.parse(BASE_VAR_FILE_PATH)
            root = var_tree.getroot()
        except OSError:
            print(f"错误：无法找到或解析模板文件 '{BASE_VAR_FILE_PATH}'。循环终止。")
            return

        # --- 3.3. 根据规则修改XML节点 ---

        # 规则1: 直接替换值 (包含单位转换)
        # 乘员体型
        dummy_map = {1: 'd_hyb305el_Q_inc.xml', 2: 'd_hyb350el_Q_inc.xml', 3: 'd_hyb395el_Q_inc.xml'}
        root.find(".//DEFINE[@VAR_NAME='dummy.dummy_type']").attrib['VALUE'] = dummy_map[occupant_type]

        # 限力值 (kN -> N)
        root.find(".//DEFINE[@VAR_NAME='load_limit.R_LL1F']").attrib['VALUE'] = str(ll1_kn * 1000.0)
        root.find(".//DEFINE[@VAR_NAME='load_limit.R_LL2F']").attrib['VALUE'] = str(ll2_kn * 1000.0)

        # 点火时刻 (ms -> s)
        root.find(".//DEFINE[@VAR_NAME='trigger_time.pretensioner_time']").attrib['VALUE'] = str(btf_ms / 1000.0)
        root.find(".//DEFINE[@VAR_NAME='trigger_time.anchor_pretensioner_time']").attrib['VALUE'] = str(ptf_ms / 1000.0)
        root.find(".//DEFINE[@VAR_NAME='trigger_time.airbag_trigger_time']").attrib['VALUE'] = str(aft_ms / 1000.0)

        # 抽入量 (mm -> m)
        root.find(".//DEFINE[@VAR_NAME='load_limit.R_LL1S']").attrib['VALUE'] = str(pp_mm / 1000.0)
        root.find(".//DEFINE[@VAR_NAME='aps.AP_S']").attrib['VALUE'] = str(plp_mm / 1000.0)

        # 二级限力切换时刻 (条件 + 单位转换 ms -> s)
        llattf_s = 2.0 if lla_status == 0 else (llattf_ms / 1000.0)
        root.find(".//DEFINE[@VAR_NAME='load_limit.R_LL2TF']").attrib['VALUE'] = str(llattf_s)

        # D环高度 (离散映射)
        d_ring_coords = {
            1: "1.823 -0.615005 0.91",
            2: "1.82603 -0.608225 0.907772",
            3: "1.829 -0.601355 0.905545",
            4: "1.83205 -0.594275 0.903251"
        }
        root.find(".//DEFINE[@VAR_NAME='belts.dring_pos']").attrib['VALUE'] = d_ring_coords[dz_level]

        # 规则2: 在默认值基础上进行增量修改
        # 座椅前后位置 (仅调整x方向)，注意正负：npz文件中加正值是座椅向前移动，而MYDYMO中加正值是座椅向后移动
        seat_pos_node = root.find(".//DEFINE[@VAR_NAME='dummy.seat_pos']")
        default_seat_pos = np.fromstring(seat_pos_node.attrib['VALUE'], sep=' ')
        new_seat_pos = default_seat_pos - np.array([sp_mm, 0, 0]) # 注意这里是减去，因为MYDYMO中加正值是座椅向后移动
        seat_pos_node.attrib['VALUE'] = ' '.join(map(str, new_seat_pos))

        # 假人位置dummy.dummy_position，同座椅位置调整
        dummy_pos_node = root.find(".//DEFINE[@VAR_NAME='dummy.dummy_position']")
        default_dummy_pos = np.fromstring(dummy_pos_node.attrib['VALUE'], sep=' ')
        new_dummy_pos = default_dummy_pos - np.array([sp_mm, 0, 0])
        dummy_pos_node.attrib['VALUE'] = ' '.join(map(str, new_dummy_pos))

        # 靠背角度及联动参数 (角度 -> 弧度)，注意正负：npz文件中加正值是靠背向后转，而MYDYMO中加正值是靠背向前转
        recline_rad = np.deg2rad(recline_deg)
        
        # 座椅整体角度
        seat_whole_node = root.find(".//DEFINE[@VAR_NAME='dummy.seat_whole_angle']")
        default_seat_whole = float(seat_whole_node.attrib['VALUE'])
        seat_whole_node.attrib['VALUE'] = str(default_seat_whole - recline_rad) # 注意这里是减去，因为MYDYMO中加正值是靠背向前转

        # 座椅平面角度
        seat_plane_node = root.find(".//DEFINE[@VAR_NAME='dummy.seat_plane_angle']")
        default_seat_plane = float(seat_plane_node.attrib['VALUE'])
        seat_plane_node.attrib['VALUE'] = str(default_seat_plane + recline_rad)

        # 假人姿态角度 (髋关节、肩关节等)，这些角度与座椅靠背角度联动
        angle_params_to_update = [
            'dummy.recline_angle', 'dummy.hip_left_angle', 'dummy.hip_right_angle',
            'dummy.shoulder_left_angle', 'dummy.shoulder_right_angle'
        ]
        for param_name in angle_params_to_update:
            node = root.find(f".//DEFINE[@VAR_NAME='{param_name}']")
            default_angle = float(node.attrib['VALUE'])
            node.attrib['VALUE'] = str(default_angle + recline_rad)

        # 规则3: 修改FUNCTION类型的值
        # 二级主动泄气孔切换
        cdt_fun_node = root.find(".//DEFINE[@VAR_NAME='trigger_time.cdt_fun']")
        if aav_status == 0:
            # 不切换
            cdt_value = "| XI YI |\n0 1\n0.5 1\n\n"
        else: # aav_status == 1
            # 切换
            ttf_s = ttf_ms / 1000.0
            cdt_value = (f"| XI YI |\n0 1\n{ttf_s:.4f} 1\n{ttf_s + 0.001:.4f} 1.5\n0.5 1.5\n\n")
        cdt_fun_node.attrib['VALUE'] = cdt_value
        #cdt_fun_node.attrib['VALUE'] = cdt_value.strip()

        # --- 3.4. 更新碰撞波形 ---
        bad_flag = False
        pulse_params = {
            'x': 'kinematic.x_input_pluse',
            'y': 'kinematic.y_input_pluse',
            'z': 'kinematic.z_rot_pluse'
        }
        for axis, var_name in pulse_params.items():
            pulse_filename = f'{axis}{case_id}.csv'
            pulse_filepath = os.path.join(PULSE_FILES_DIR, pulse_filename)
            
            try:
                # 使用numpy加载数据，假设CSV格式为 "时间,加速度" 或 "时间 加速度"
                # np.loadtxt能很好地处理这两种情况，并自动跳过带#的注释行
                time_ori = np.loadtxt(pulse_filepath)[:, 0] 
                acceleration_values = np.loadtxt(pulse_filepath)[:, 1]
                
                # 获取数据点的总数
                num_points = len(acceleration_values)
                
                # 自行生成精确的时间戳;
                dt = np.mean(np.diff(time_ori))
                if np.isclose(dt, 5e-6, atol=1e-7):
                    timestamps = np.arange(num_points) * 5e-6
                elif np.isclose(dt, 1e-5, atol=1e-7):
                    timestamps = np.arange(num_points) * 1e-5
                else:
                    raise ValueError(f"时间间隔 {dt} 不符合预期的5e-6或1e-5秒。")

                # 准备符合MADYMO格式的多行字符串
                # 1. 添加头部
                formatted_lines = "	|	XI	YI	|\n"
                
                # 2. 循环生成每一行 "时间戳 加速度值"
                #    使用格式化字符串确保不出现科学计数法，并保留足够的小数位数
                lines_list = []
                for ts, acc in zip(timestamps, acceleration_values):
                    # 加速度值保留8位小数，中间用制表符分隔
                    if dt == 5e-6:
                        lines_list.append(f"{ts:.6f}\t{acc:.8f}")
                    elif dt == 1e-5:
                        lines_list.append(f"{ts:.5f}\t{acc:.8f}")
                    else:
                        lines_list.append(f"{ts:.7f}\t{acc:.8f}")
                
                # 3. 将所有行合并成一个字符串
                formatted_lines += "\n".join(lines_list)
                formatted_lines += "\n"  
                
                pulse_node = root.find(f".//DEFINE[@VAR_NAME='{var_name}']")
                pulse_node.attrib['VALUE'] = formatted_lines
                
            except FileNotFoundError:
                bad_flag = True
                print(f"  - 警告：未找到碰撞波形文件 '{pulse_filepath}'，直接跳过写入最终的.var文件。")
            except IndexError:
                bad_flag = True
                print(f"  - 错误：无法从 '{pulse_filepath}' 中读取加速度数据。请检查文件格式是否正确（应至少有两列）。，直接跳过写入最终的.var文件。")
            except Exception as e:
                bad_flag = True
                print(f"  - 错误：处理文件 '{pulse_filepath}' 时出错: {e}，直接跳过写入最终的.var文件。")

        # --- 3.5. 写入最终的.var文件 ---
        if not bad_flag:
            output_filename = f'data_{case_id}.var'
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)
            
            # 步骤 A: 先使用lxml正常写入文件，此时换行符和制表符会被转义
            tree = etree.ElementTree(root)
            tree.write(output_filepath, pretty_print=True, xml_declaration=True, encoding='UTF-8')

            # 步骤 B: 后处理文件，将字符实体替换为实际字符
            try:
                # 以二进制模式读取，防止编码问题
                with open(output_filepath, 'rb') as f:
                    content_bytes = f.read()
                
                # 将字节解码为字符串
                content_str = content_bytes.decode('utf-8')
                
                # 执行替换操作
                content_str = content_str.replace('&#9;', '\t')   # 将Tab实体替换为制表符
                content_str = content_str.replace('&#10;', '\n')  # 将Newline实体替换为换行符
                
                # 将修正后的内容写回文件
                with open(output_filepath, 'wb') as f:
                    f.write(content_str.encode('utf-8'))

            except Exception as e:
                print(f"  - 错误: 在后处理文件 '{output_filepath}' 时发生错误: {e}")

    print(f"\n处理完成！所有.var文件已生成至 '{OUTPUT_DIR}' 目录，并已完成格式修正。")

if __name__ == '__main__':
    generate_var_files()