# -*- coding: utf-8 -*-
"""
MADYMO XML 自动化生成脚本 (Step 3 - 20260302 版本)
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
from datetime import datetime

# ==================== 1. 全局配置 ====================

# --- 指定本次运行要生成的 XML 类型 ---
# 格式: '{驾驶侧}_{假人体型}'
# 可选: 'DS_5th', 'DS_50th', 'DS_95th', 'PS_5th', 'PS_50th', 'PS_95th'
# =================================*****===================================
XML_TYPE = 'PS_5th'
# =================================*****===================================
# --- 路径配置 ---
# 请根据实际环境确认以下路径
BASE_XML_DIR = r'D:\MADYMO_xml文件'
# =================================*****===================================
PARAM_FILE_PATH = r'I:\000 LX\dataset0715\03\distribution_0302.csv'
# =================================*****===================================
PULSE_FILES_DIR = r'I:\000 LX\dataset0715\03\acc_data_before1111_6134'
OUTPUT_DIR = os.path.join(BASE_XML_DIR, XML_TYPE)

# Summary CSV (per-run, timestamped). Only "batch" mode implemented (one file per run).
SUMMARY_CSV_FILENAME_TEMPLATE = 'xml_generation_summary_{ts}.csv'
SUMMARY_CSV_MODE = 'batch'  # future: 'append' supported if needed

# --- Base XML 文件映射 ---
# =================================*****===================================
BASE_XML_PATHS = {
    'DS_5th':  os.path.join(BASE_XML_DIR, '主驾-5th假人-base-V6-0225.xml'),
    'DS_50th': os.path.join(BASE_XML_DIR, '主驾-50th假人-base-V5-0121.xml'),
    'DS_95th': os.path.join(BASE_XML_DIR, '主驾-95th假人-base-V6-0225.xml'),
    'PS_5th':  os.path.join(BASE_XML_DIR, '副驾-5th假人-base-0104.xml'),
    'PS_50th': os.path.join(BASE_XML_DIR, '副驾-50th假人-base-1221.xml'),
    'PS_95th': os.path.join(BASE_XML_DIR, '副驾-95th假人-base-1226.xml'),
}
# =================================*****===================================
# --- 常量配置 ---
CASE_ID_LIST = None        # 指定特定case_id列表 (None表示处理所有)
# CASE_ID_LIST = [2, 50001, 50004, 50005] # 调试用

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

def calc_joint_angles(ot, is_driver, Seat_X_Disp, Seat_Z_Disp, Seat_Back_rotation_Angle):
    """
    根据xml文件座椅位置 Seat_X_Disp(m) , 座椅高度Seat_Z_Disp(m) , 座椅靠背角度Seat_Back_rotation_Angle(rad) 和假人类型 OT 计算关节角度(rad)
    """
    if is_driver == 0: # 副驾PS
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
                
    if is_driver == 1: # 主驾DS
        # 初始化8个假人关节角度
        hipL, hipR, AnkleL, AnkleR, elbow, shoulder, kneeL, kneeR = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # --- 主驾5th假人 (OT=1) ---
        if ot == 1:
            if Seat_Z_Disp <= 0.03:

                x = Seat_X_Disp / 0.1
                z = Seat_Z_Disp / 0.1
                A = Seat_Back_rotation_Angle
                sinA = np.sin(A)
                cosA = np.cos(A)                

                # (1) hipL_angle   R²=0.943
                hipL = (
                    -0.259212
                    - 1.436109*x - 0.414419*z + 6.555674*x*z
                    - 1.666517*sinA + 0.363335*cosA
                    + 0.607384*x*sinA + 1.319137*x*cosA
                    + 2.431542*z*sinA + 0.982561*z*cosA
                    - 2.480724*x*z*sinA - 6.811193*x*z*cosA
                )

                # (2) kneeL_angle  R²=0.802
                kneeL = (
                    2.990427
                    - 1.118961*x + 29.925925*z - 19.203624*x*z
                    + 0.528269*sinA - 2.907787*cosA
                    - 0.311200*x*sinA + 1.238635*x*cosA
                    - 7.883337*z*sinA - 31.191876*z*cosA
                    + 7.230564*x*z*sinA + 19.878176*x*z*cosA
                )

                # (3) AnkleL_angle R²=0.764
                AnkleL = (
                    7.093497
                    - 1.892864*x - 187.764977*z + 119.438948*x*z
                    + 0.487504*sinA - 6.425926*cosA
                    - 1.049344*x*sinA + 1.523374*x*cosA
                    + 2.693904*z*sinA + 188.035787*z*cosA
                    - 1.784960*x*z*sinA - 119.509369*x*z*cosA
                )

                # (4) hipR_angle   R²=0.998
                hipR = (
                    3.980650
                    - 3.558642*x - 5.056427*z + 5.320563*x*z
                    - 1.577204*sinA - 3.711162*cosA
                    + 0.310685*x*sinA + 3.354325*x*cosA
                    + 0.418618*z*sinA + 5.426328*z*cosA
                    - 0.375501*x*z*sinA - 5.386509*x*z*cosA
                )

                # (5) kneeR_angle  R²=0.997
                kneeR = (
                    -6.864789
                    + 5.863100*x + 21.445177*z - 18.562829*x*z
                    + 1.154641*sinA + 6.331442*cosA
                    - 0.513115*x*sinA - 5.330949*x*cosA
                    - 1.338176*z*sinA - 21.987889*z*cosA
                    + 1.176342*x*z*sinA + 18.790870*x*z*cosA
                )

                # (6) AnkleR_angle R²=0.990
                AnkleR = (
                    -0.722018
                    + 1.288186*x - 5.281417*z + 6.344441*x*z
                    - 0.281222*sinA + 1.367993*cosA
                    - 0.019368*x*sinA - 1.657844*x*cosA
                    - 0.875387*z*sinA + 5.364009*z*cosA
                    + 0.443106*x*z*sinA - 6.438599*x*z*cosA
                )

                # (7) shoulder_angle R²=0.958
                shoulder = (
                    5.387519
                    - 6.272059*x + 105.990980*z - 80.865044*x*z
                    + 2.170030*sinA - 6.756160*cosA
                    - 0.593709*x*sinA + 6.626770*x*cosA
                    - 10.168768*z*sinA - 105.863349*z*cosA
                    + 7.939414*x*z*sinA + 80.784567*x*z*cosA
                )

                # (8) elbow_angle   R²=0.987
                elbow = (
                    -6.458139
                    + 9.050801*x - 207.536031*z + 151.396515*x*z
                    - 5.548801*sinA + 6.743567*cosA
                    + 1.054507*x*sinA - 9.826591*x*cosA
                    + 18.270903*z*sinA + 207.764405*z*cosA
                    - 13.692283*x*z*sinA - 151.462539*x*z*cosA
                )

                return hipL, hipR, kneeL, kneeR, AnkleL, AnkleR, shoulder, elbow

            elif Seat_Z_Disp > 0.03:
                # X=Seat_X_Disp, Z=Seat_Z_Disp, A=Seat_Back_rotation_Angle (rad)
                # 无量纲化：把范围大致映射到 [-1, 1]
                x = (Seat_X_Disp - 0.135) / 0.015
                z = (Seat_Z_Disp - 0.045) / 0.015
                A = Seat_Back_rotation_Angle
                a = (A - 0.0673) / 0.0873
                sinA = np.sin(A)
                cosA = np.cos(A)

                hipL = (
                    -0.01991706545
                    + (0.02593918565)*z
                    + (-0.06112241668)*a
                    + (-4.613893937e-05)*(x**2)
                    + (0.009909408019)*(z**2)
                    + (0.004853243092)*(a**2)
                    + (0.01238168779)*(z*a)
                    + (-0.006657797283)*sinA
                    + (-0.01953203253)*cosA
                    + (0.002921751205)*(z*sinA)
                    + (0.02729531192)*(z*cosA)
                )

                kneeL = (
                    0.03744247518
                    + (-0.0468954927)*z
                    + (-0.01233599623)*a
                    + (0.007902090308)*(x**2)
                    + (0.02395964305)*(z**2)
                    + (0.008663217206)*(a**2)
                    + (-0.03976882437)*(z*a)
                    + (0.001442627947)*sinA
                    + (0.03739712737)*cosA
                    + (-0.006605872921)*(z*sinA)
                    + (-0.04644677034)*(z*cosA)
                )

                AnkleL = (
                    0.09068266463
                    + (0.01636727423)*z
                    + (-0.1003841563)*a
                    + (0.005452635647)*(x**2)
                    + (-0.04158453552)*(z**2)
                    + (-0.0009211460836)*(a**2)
                    + (0.007041118227)*(z*a)
                    + (-0.002634022612)*sinA
                    + (0.09106947069)*cosA
                    + (0.00171402576)*(z*sinA)
                    + (0.01630110551)*(z*cosA)
                )

                hipR = (
                    0.02910819225
                    + (-0.05600975897)*x
                    + (-0.003134475853)*z
                    + (-0.1007329814)*a
                    + (0.003201656335)*(x**2)
                    + (-0.01666532209)*(z**2)
                    + (-5.508989903e-05)*(a**2)
                    + (0.004196973235)*(x*z)
                    + (-2.263755205e-05)*(x*a)
                    + (-0.0009924370478)*(z*a)
                    + (-0.006805423825)*sinA
                    + (0.02963314463)*cosA
                    + (0.002551171854)*(x*sinA)
                    + (0.003383647811)*(z*sinA)
                    + (0.03787955817)*(x*cosA)
                    + (0.05148785825)*(z*cosA)
                )

                kneeR = (
                    0.02601430851
                    + (0.3217359568)*x
                    + (0.05400333035)*z
                    + (0.04035963954)*a
                    + (-0.01466694492)*(x**2)
                    + (0.04416623753)*(z**2)
                    + (-0.001233949883)*(a**2)
                    + (-0.01827078537)*(x*z)
                    + (-0.0005885848577)*(x*a)
                    + (-0.00148867129)*(z*a)
                    + (0.00526071432)*sinA
                    + (0.02572346134)*cosA
                    + (-0.01839343096)*(x*sinA)
                    + (-0.007849379154)*(z*sinA)
                    + (-0.2721292105)*(x*cosA)
                    + (-0.1145264614)*(z*cosA)
                )

                AnkleR = (
                    0.03123648952
                    + (-0.03375536582)*x
                    + (-0.008647576774)*z
                    + (-0.0259724087)*a
                    + (0.004922459341)*(x**2)
                    + (0.02770747718)*(z**2)
                    + (0.0009116757959)*(a**2)
                    + (-0.01082585496)*(x*z)
                    + (0.001889666346)*(x*a)
                    + (-0.001642927586)*(z*a)
                    + (-0.0001589902212)*sinA
                    + (0.03131459931)*cosA
                    + (-0.002100030653)*(x*sinA)
                    + (-0.0007173785596)*(z*sinA)
                    + (-0.03360686467)*(x*cosA)
                    + (-0.008513246749)*(z*cosA)
                )

                shoulder = (
                    -0.3952390938
                    + (0.009193961452)*x
                    + (0.001428946862)*z
                    + (0.1067668728)*a
                    + (0.01316775698)*(x**2)
                    + (-0.000576327751)*(z**2)
                    + (-0.002050197538)*(a**2)
                    + (-0.007869362243)*(x*z)
                    + (0.008703500911)*(x*a)
                    + (0.00982118854)*(z*a)
                    + (-0.01729115124)*sinA
                    + (-0.3949625824)*cosA
                    + (0.001361573794)*(x*sinA)
                    + (0.0008697086685)*(z*sinA)
                    + (0.008916693986)*(x*cosA)
                    + (0.0001700945729)*(z*cosA)
                )

                elbow = (
                    -0.44319821
                    + (-0.02708214276)*x
                    + (0.01429528143)*z
                    + (-0.3206224287)*a
                    + (-0.06415023316)*(x**2)
                    + (-0.01239120881)*(z**2)
                    + (0.0008869934282)*(a**2)
                    + (0.00296014373)*(x*z)
                    + (-0.02033895738)*(x*a)
                    + (-0.02477991171)*(z*a)
                    + (-0.05769646978)*sinA
                    + (-0.4403183344)*cosA
                    + (-0.004534417337)*(x*sinA)
                    + (-0.000535816287)*(z*sinA)
                    + (-0.04090480757)*(x*cosA)
                    + (0.02417777455)*(z*cosA)
                )

            return hipL, hipR, kneeL, kneeR, AnkleL, AnkleR, shoulder, elbow

        # --- 主驾50th假人 (OT=2) ---
        elif ot == 2:
            # 根据公式要求，对X和Z进行归一化处理: x = X/0.1, z = Z/0.1
            x = Seat_X_Disp / 0.1
            z = Seat_Z_Disp / 0.1
            A = Seat_Back_rotation_Angle
            sinA = np.sin(A)
            cosA = np.cos(A)
            
            # 1) hipL_angle
            hipL = (-3.3573 + 4.7475*x - 6.8897*z 
                    - 1.2473*sinA + 3.4861*cosA 
                    + 0.0702*x*sinA - 5.1564*x*cosA 
                    - 0.1564*z*sinA + 7.1326*z*cosA)
            
            # 2) kneeL_angle
            kneeL = (3.2752 - 4.9081*x + 3.7214*z 
                     + 0.2760*sinA - 3.6008*cosA 
                     + 0.1346*x*sinA + 5.8424*x*cosA 
                     + 0.1545*z*sinA - 3.8715*z*cosA)
            
            # 3) AnkleL_angle
            AnkleL = (3.5722 - 3.9128*x - 1.0510*z 
                      + 0.0780*sinA - 3.5596*cosA 
                      - 0.2896*x*sinA + 3.7978*x*cosA 
                      - 0.1528*z*sinA + 1.0101*z*cosA)
            
            # 4) hipR_angle
            hipR = (-5.6526 + 3.5477*x + 4.6278*z 
                    - 1.4147*sinA + 5.7760*cosA 
                    + 0.1167*x*sinA - 3.8220*x*cosA 
                    + 0.2219*z*sinA - 4.4373*z*cosA)
            
            # 5) kneeR_angle
            kneeR = (1.2849 - 2.9211*x + 3.3403*z 
                     + 0.3666*sinA - 1.5224*cosA 
                     + 0.0528*x*sinA + 3.5022*x*cosA 
                     + 0.0819*z*sinA - 3.5360*z*cosA)
            
            # 6) AnkleR_angle
            AnkleR = (0.0510 - 0.2372*x - 1.9154*z 
                      - 0.1014*sinA - 0.0182*cosA 
                      - 0.1907*x*sinA - 0.0876*x*cosA 
                      - 0.1604*z*sinA + 1.8528*z*cosA)
            
            # 7) shoulder_angle
            shoulder = (-1.5157 + 5.6963*x - 8.2674*z 
                        - 0.5847*sinA + 0.6854*cosA 
                        + 1.9682*x*sinA - 5.4041*x*cosA 
                        - 0.2962*z*sinA + 8.3978*z*cosA)
            
            # 8) elbow_angle
            elbow = (2.1810 - 6.4882*x + 11.1580*z 
                     - 1.3810*sinA - 2.7929*cosA 
                     - 1.9498*x*sinA + 5.8832*x*cosA 
                     + 0.6987*z*sinA - 11.1209*z*cosA)
            
            return hipL, hipR, kneeL, kneeR, AnkleL, AnkleR, shoulder, elbow
            
        # --- 主驾95th假人 (OT=3) ---
        elif ot == 3:
            # ===== dimensionless variables (from your dataset) =====
            A = Seat_Back_rotation_Angle  # (rad)
            # X = Seat_X_Disp (length), Z = Seat_Z_Disp (length), A = Seat_Back_rotation_Angle (radian)
            x = (Seat_X_Disp - 0.025) / 0.03335804299481097
            z = (Seat_Z_Disp - 0.03)  / 0.04447739065974796
            a = (A - 0.02365) / 0.09707190511489994
            sina = np.sin(a)
            cosa = np.cos(a)

            # ===== explicit formulas (8 joint angles) =====
            hipL = -0.0162762914008 - (0.0713251836636)*x + (0.0956070491711)*z - (0.13078403719)*a + (0.0063550382552)*(x**2) - (0.000637671164168)*(z**2) - (0.00777245318083)*(a**2) - (0.00442742142735)*(x*z) - (0.000177181065433)*(x*a) - (0.00442742142735)*(z*a) + (0.021645713508)*sina - (0.012772744104)*cosa

            kneeL= 0.0298788396842 + (0.174346718093)*x - (0.0777145905368)*z + (0.021554160023)*a + (0.00414872725375)*(x**2) + (0.0109842488511)*(z**2) + (0.0186813997208)*(a**2) + (0.00558103455845)*(x*z) + (0.00906265742691)*(x*a) + (0.00483688704827)*(z*a) + (0.00654416241494)*sina + (0.0215903760261)*cosa

            AnkleL = -0.102526793842 - (0.111136464177)*x - (0.00606939261335)*z - (0.00899187932414)*a + (0.00511351922666)*(x**2) - (0.0165962944655)*(z**2) - (0.0384792349214)*(a**2) + (0.0015436469412)*(x*z) - (0.000335240390053)*(x*a) - (0.00077182662978)*(z*a) - (0.00880890522167)*sina - (0.0848671689378)*cosa

            hipR = -0.013563644713 - (0.0598497666178)*x + (0.0910390284829)*z - (0.135281868829)*a + (0.00533191354579)*(x**2) - (0.00473949883539)*(z**2) - (0.00462593229458)*(a**2) + (0.00113335564863)*(x*z) + (0.00493322300309)*(x*a) - (0.00319400660047)*(z*a) + (0.0287430296176)*sina - (0.0114228787585)*cosa

            kneeR = 0.0392884443502 + (0.153562790832)*x - (0.0718355286652)*z + (0.0246036714958)*a + (0.000177495602311)*(x**2) + (0.000547302314585)*(z**2) + (0.0184028310976)*(a**2) + (0.00847079994158)*(x*z) - (0.00195487842845)*(x*a) + (0.00534490169852)*(z*a) + (0.0150251951003)*sina + (0.0309823301374)*cosa

            AnkleR = -0.110800035015 - (0.0874645334181)*x - (0.00916633415458)*z + (0.0122137343596)*a + (0.00840756384742)*(x**2) - (0.0253225876562)*(z**2) - (0.0391336354988)*(a**2) + (0.00285419810093)*(x*z) - (0.00332753393492)*(x*a) + (0.00366968033521)*(z*a) - (0.0445331407416)*sina - (0.0927465119968)*cosa

            shoulder = -0.363258120794 + (0.0904271767945)*x + (0.0862720025071)*z + (0.0358019638482)*a - (0.00584321602232)*(x**2) - (0.0310372415097)*(z**2) - (0.115971197736)*(a**2) + (0.0553785815594)*(x*z) + (0.00686631424188)*(x*a) - (0.00405208722575)*(z*a) + (0.0316931917699)*sina - (0.309256812773)*cosa

            elbow = -0.530287761902 - (0.196405799741)*x - (0.0108233838651)*z - (0.198083103126)*a + (0.00676874738226)*(x**2) - (0.0327186052733)*(z**2) - (0.20853709902)*(a**2) - (0.0726315302737)*(x*z) + (0.0195822727847)*(x*a) + (0.0329048937916)*(z*a) - (0.0912069082986)*sina - (0.434945286464)*cosa
            return hipL, hipR, kneeL, kneeR, AnkleL, AnkleR, shoulder, elbow


def get_pulse_data_string(pulse_dir, pulse_case_id, axis):
    """
    读取CSV波形并格式化为 MADYMO XY_PAIR 字符串
    """
    csv_path = os.path.join(pulse_dir, f'{axis}{pulse_case_id}.csv')
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
        val_sh_m  = get_val(['SH']) / 1000.0           # mm -> m 
        
        # 原始值读取
        raw_llattf = get_val(['LLATTF'])               # ms 
        raw_ra_deg = get_val(['RA'])                   # degree 
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
    delta_ra_rad = -np.deg2rad(raw_ra_deg - 25.0) # 为正值时座椅靠背相对中立位前倾

    # DZ 逻辑 
    val_dring_pos = DRING_COORDS_MAP[is_driver][dz_level]

    # --- 2. 修改 XML define 变量 ---

    # 单独处理 Seat_X_Disp 和 Seat_Back_rotation_Angle，因为要获取base value；且 Seat_X_Disp 影响关节角度计算
    # 处理 Seat_X_Disp = base value + SP采样值
    sp_nodes = root.xpath(".//DEFINE[@VAR_NAME='Seat_X_Disp']")
    if sp_nodes:
        base_sp_val = float(sp_nodes[0].attrib['VALUE'])
        Seat_X_Disp = base_sp_val + val_sp_m # 单位: m
        sp_nodes[0].attrib['VALUE'] = f"{Seat_X_Disp:.6f}"
    else:
        raise ValueError(f"  [Error] Case {case_id}: Base XML 中未找到变量 Seat_X_Disp")

    # 处理 Seat_Z_Disp = base value + SH采样值
    # 只有主驾（is_driver==1）会用到高度参数
    if is_driver == 1:
        sz_nodes = root.xpath(".//DEFINE[@VAR_NAME='Seat_Z_Disp']")
        if sz_nodes:
            base_sz_val = float(sz_nodes[0].attrib['VALUE'])
            Seat_Z_Disp = base_sz_val + val_sh_m # 座椅高度, 单位: m
            sz_nodes[0].attrib['VALUE'] = f"{Seat_Z_Disp:.6f}"
        else:
            raise ValueError(f"  [Error] Case {case_id}: Base XML 中未找到变量 Seat_Z_Disp")
    else:
        # 非司机侧不使用高度，赋0作为占位
        Seat_Z_Disp = 0.0

    # Seat_Back_rotation_Angle = base value - deg2rad[RA采样值 - 25°]
    ra_nodes = root.xpath(".//DEFINE[@VAR_NAME='Seat_Back_rotation_Angle']")
    if ra_nodes:
        base_ra_val = float(ra_nodes[0].attrib['VALUE'])
        Seat_Back_rotation_Angle = base_ra_val + delta_ra_rad # 注意这里是加上 delta，因为 delta 已经包含了负号; 单位: rad
        ra_nodes[0].attrib['VALUE'] = f"{Seat_Back_rotation_Angle:.6f}"
    else:
        raise ValueError(f"  [Error] Case {case_id}: Base XML 中未找到变量 Seat_Back_rotation_Angle")
    
    # 三个关节角度计算（基于更新后的 Seat_X_Disp, Seat_Z_Disp, Seat_Back_rotation_Angle）
    # 目前副驾时 Seat_Z_Disp 的值不会参与计算
    joint_angles = calc_joint_angles(ot, is_driver, Seat_X_Disp, Seat_Z_Disp, Seat_Back_rotation_Angle)
    
    # 根据驾驶侧解包角度值
    if is_driver == 0:  # 副驾：3个角度 (hip, knee, ankle)
        val_hip, val_knee, val_ankle = joint_angles
    else:  # 主驾：8个角度
        val_hipL, val_hipR, val_kneeL, val_kneeR, val_AnkleL, val_AnkleR, val_shoulder, val_elbow = joint_angles

    # 定义其它的要修改的变量映射 {VAR_NAME: New_Value_String}
    if is_driver == 0:  # 副驾
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
    elif is_driver == 1:  # 主驾
        vars_map = {
            "R_LL1F": f"{val_ll1_N:.6f}",       # 一级限力值
            "R_LL2F": f"{val_ll2_N:.6f}",       # 二级限力值
            "RPTTF_def": f"{val_btf_s:.6f}",    # 预紧器点火
            "APTTF_def": f"{val_ptf_s:.6f}",    # 腰部预紧器点火
            "R_LL2TF": f"{val_ll2tf_s:.6f}",    # 二级限力切换
            "Dring_pos": val_dring_pos,         # D环位置
            "DAB_TTF": f"{val_aft_s:.6f}",      # 气囊点火,注意主驾是DAB!!!
            "hipL_angle": f"{val_hipL:.6f}",    # 八个关节角度
            "hipR_angle": f"{val_hipR:.6f}",
            "kneeL_angle": f"{val_kneeL:.6f}",
            "kneeR_angle": f"{val_kneeR:.6f}",
            "AnkleL_angle": f"{val_AnkleL:.6f}",
            "AnkleR_angle": f"{val_AnkleR:.6f}",
            "shoulder_angle": f"{val_shoulder:.6f}",
            "elbow_angle": f"{val_elbow:.6f}"
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
    
    # 波形文件来源由 pulse_source_case_id 显式指定
    if 'pulse_source_case_id' not in case_data:
        raise ValueError(f"  [Error] Case {case_id}: 缺少必要列 pulse_source_case_id")
    try:
        pulse_case_id = int(case_data['pulse_source_case_id'])
    except Exception as e:
        raise ValueError(f"  [Error] Case {case_id}: pulse_source_case_id 非法 ({case_data['pulse_source_case_id']})") from e

    pulse_configs = [
        {'axis': 'x', 'exact_name': 'X_lin_pulse_fun'},
        {'axis': 'y', 'exact_name': 'Y_lin_pulse_fun'}
    ]

    for p_conf in pulse_configs:
        try:
            # 读取并格式化波形数据
            pulse_str = get_pulse_data_string(PULSE_FILES_DIR, pulse_case_id, p_conf['axis'])
            
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
    print(f"使用 Base XML 模板: {base_xml_path}")

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

    required_cols = ['pulse_source_case_id', 'is_driver_side', 'OT', 'is_pulse_ok', 'is_injury_ok']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"参数文件缺少必要列: {missing_cols}")

    # ------------ summary-CSV 初始化 (严格列顺序，仅记录成功生成的 XML) ------------
    # first: case_id; next guaranteed-from-distribution: impact_velocity, impact_angle, overlap
    summary_distribution_cols = ['impact_velocity','impact_angle','overlap','LL1','LL2','BTF','LLATTF','PTF','AFT','SP','SH','RA','DZ','is_driver_side','OT','pulse_source_case_id']
    summary_varname_cols = ['Seat_X_Disp','Seat_Z_Disp','Seat_Back_rotation_Angle','R_LL1F','R_LL2F','RPTTF_def','APTTF_def','R_LL2TF','Dring_pos','DAB_TTF','PAB_TTF','hip_angle','knee_angle','ankle_angle','hipL_angle','hipR_angle','kneeL_angle','kneeR_angle','AnkleL_angle','AnkleR_angle','shoulder_angle','elbow_angle']
    summary_columns = ['case_id'] + summary_distribution_cols + summary_varname_cols
    summary_rows = []
    # -------------------------------------------------------------------------------
    
    # 3. 筛选要处理的 Case
    # 条件：
    # (1) 对应侧 (主驾或副驾)
    # (2) 对应假人类型 (OT)
    # (3) Pulse OK
    # (4) Injury Not OK (只跑还没跑过的或失败的)
    
    # 获取波形数据目录下的所有的波形case_id
    valid_pulse_ids = []
    if os.path.exists(PULSE_FILES_DIR):
        for f in os.listdir(PULSE_FILES_DIR):
            if f.startswith('x') and f.endswith('.csv'):
                try:
                    valid_pulse_ids.append(int(f[1:-4]))
                except:
                    pass
    valid_pulse_ids = set(valid_pulse_ids)

    # 构造待处理列表
    tasks = []
    for cid in df.index:
        # 如果指定了 list，则只处理 list 中的
        if CASE_ID_LIST is not None and cid not in CASE_ID_LIST:
            continue

        row = df.loc[cid]
        
        # 检查是否为主副驾（严格要求来自列值）
        current_is_driver = int(row['is_driver_side'])
        
        # 匹配配置的驾驶侧
        if current_is_driver != is_driver_side:
            continue
            
        # 匹配假人 OT（严格要求来自 OT 列）
        row_ot = row['OT']
        if row_ot != ot:
            continue
            
        # 检查波形是否可用的标志位, 必须为True
        if row.get('is_pulse_ok') != True:
            continue
            
        # 检查波形文件是否存在（严格由 pulse_source_case_id 指定）
        pulse_case_id = int(row['pulse_source_case_id'])
        if pulse_case_id not in valid_pulse_ids:
            continue

        ###################################################################
        # is_injury_ok为true或false都不处理
        if row.get('is_injury_ok') == True or row.get('is_injury_ok') == False:
            continue
        # 仅处理caseid>70000的工况
        if cid <= 70000:
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

                # --- 记录 summary 行（仅在成功写入后） ---
                try:
                    # reuse the tree we just modified instead of reparsing the written file,
                    # avoids lxml 'huge input lookup' errors when XML contains large CDATA.
                    parsed_out = new_tree

                    def _get_define(name):
                        nodes = parsed_out.xpath(f".//DEFINE[@VAR_NAME='{name}']")
                        if not nodes:
                            return ''
                        return nodes[0].attrib.get('VALUE', '')

                    row = {'case_id': case_id}
                    # distribution-sourced columns (preserve names)
                    src = df.loc[case_id]
                    for c in summary_distribution_cols:
                        # prefer exact column, fallback to empty string
                        row[c] = (src[c] if c in src.index else '')

                    # VAR_NAME columns read from the written XML (string values)
                    for vn in summary_varname_cols:
                        row[vn] = _get_define(vn)

                    summary_rows.append(row)
                except Exception as _err:
                    # do NOT fail the whole run for summary collection; warn and continue
                    print(f"  [Warning] Case {case_id}: 无法记录 summary 行: {_err}")

                success_count += 1
                if success_count % 50 == 0:
                    print(f"已生成 {success_count} 个文件...")
        except Exception as e:
            print(f"Case {case_id} 处理发生未知异常: {e}")

    # 在结束前批量写入 summary CSV（原子替换 tmp -> final）
    if summary_rows:
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        summary_name = SUMMARY_CSV_FILENAME_TEMPLATE.format(ts=ts)
        summary_path = os.path.join(OUTPUT_DIR, summary_name)
        tmp_path = summary_path + '.tmp'
        try:
            pd.DataFrame(summary_rows, columns=summary_columns).to_csv(tmp_path, index=False)
            os.replace(tmp_path, summary_path)
            print(f"Summary CSV 写入: {summary_path}")
        except Exception as _err:
            print(f"[Warning] 无法写入 summary CSV: {_err}")

    print("="*60)
    print(f"处理完成！")
    print(f"成功: {success_count} / 总任务: {len(tasks)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)

if __name__ == '__main__':
    generate_xml_files()