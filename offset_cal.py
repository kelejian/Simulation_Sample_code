# %%
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Any

# ----------------------------------------------------------------------------
# 核心计算函数
# ----------------------------------------------------------------------------
def calculate_x_offset(
    wall_center: Tuple[float, float],
    wall_L: float,
    wall_W: float,
    wall_theta_deg: float,
    car_center: Tuple[float, float],
    car_l: float,
    car_l1: float,
    car_alpha_deg: float,
) -> Tuple[float, Tuple[float, float]]:
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
            min_gap_point_y = y_c
            
    # 计算最小间隙点在墙上的坐标
    if abs(y_tr - y_br) < 1e-9:
        x_wall_at_min = max(x_tr, x_br)
    else:
        x_wall_at_min = x_br + (x_tr - x_br) * (min_gap_point_y - y_br) / (y_tr - y_br)
    
    min_gap_point_on_wall = (x_wall_at_min, min_gap_point_y)

    x_offset = min_gap

    return x_offset, min_gap_point_on_wall

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

# ----------------------------------------------------------------------------
# 可视化函数用以验证
# ----------------------------------------------------------------------------
def run_and_visualize_case(case_name: str, params: Dict[str, Any], save_path: str = None) -> None:
    """运行单个测试用例并生成可视化图表"""
    
    # 解包参数
    wall_params = params["wall"]
    car_params = params["car"]

    # 计算 x_offset
    x_offset, min_gap_point_on_wall = calculate_x_offset(
        wall_center=wall_params["center"],
        wall_L=wall_params["L"],
        wall_W=wall_params["W"],
        wall_theta_deg=wall_params["theta_deg"],
        car_center=car_params["center"],
        car_l=car_params["l"],
        car_l1=car_params["l1"],
        car_alpha_deg=car_params["alpha_deg"],
    )

    print(f"--- Case: {case_name} ---")
    if math.isinf(x_offset):
        print("**警告: x_offset为inf, 无法画图")
        return
    print(f"Calculated x_offset = {x_offset:.4f}")
    print(f"Min gap occurs at y = {min_gap_point_on_wall[1]:.4f}")
    
    # --- 准备绘图数据 ---
    
    # 1. 墙的右侧边 (平移前)
    x1, y1 = wall_params["center"]
    theta_rad = math.radians(wall_params["theta_deg"])
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    
    p_tr_local = (wall_params["W"] / 2, wall_params["L"] / 2)
    p_br_local = (wall_params["W"] / 2, -wall_params["L"] / 2)
    
    x_tr_rot = p_tr_local[0] * cos_t - p_tr_local[1] * sin_t + x1
    y_tr_rot = p_tr_local[0] * sin_t + p_tr_local[1] * cos_t + y1
    x_br_rot = p_br_local[0] * cos_t - p_br_local[1] * sin_t + x1
    y_br_rot = p_br_local[0] * sin_t + p_br_local[1] * cos_t + y1
    
    wall_x_before = [x_br_rot, x_tr_rot]
    wall_y = [y_br_rot, y_tr_rot]
    
    # 2. 墙的右侧边 (平移后)
    wall_x_after = [x + x_offset for x in wall_x_before]

    # 3. 车的左边界
    x2, y2 = car_params["center"]
    l, l1 = car_params["l"], car_params["l1"]
    y_car_min, y_car_max = y2 - l / 2, y2 + l / 2
    y_car_top_corner = y2 + l1 / 2
    y_car_bottom_corner = y2 - l1 / 2
    
    y_car_points = sorted([y_car_min, y_car_bottom_corner, y_car_top_corner, y_car_max])
    x_car_points = [get_x_car_left(y, car_params) for y in y_car_points]

    # --- 开始绘图 ---
    plt.figure(figsize=(10, 8))
    #plt.style.use('seaborn-v0_8-whitegrid')
    
    # 绘制平移前的墙
    plt.plot(wall_x_before, wall_y, 'r--', label=f'Wall (before shift, θ={wall_params["theta_deg"]}°)')
    
    # 绘制平移后的墙
    plt.plot(wall_x_after, wall_y, 'r-', lw=1.0, label='Wall (after shift)')
    
    # 绘制车
    plt.plot(x_car_points, y_car_points, 'b-', lw=1.0, label=f'Car (α={car_params["alpha_deg"]}°)')
    
    # 标记最小距离点
    x_wall_at_min, y_at_min = min_gap_point_on_wall
    x_car_at_min = x_wall_at_min + x_offset
    
    plt.plot([x_wall_at_min, x_car_at_min], [y_at_min, y_at_min], 'k--')
    plt.scatter([x_wall_at_min, x_car_at_min], [y_at_min, y_at_min], c='black', zorder=5, s=10)  # 小点尺寸

    # 添加箭头和文本标注
    mid_x = (x_wall_at_min + x_car_at_min) / 2
    plt.annotate(
        f'Δx = {x_offset:.3f}',
        xy=(x_wall_at_min, y_at_min),
        xytext=(mid_x, y_at_min + 0.5),
        arrowprops=dict(arrowstyle='<->', color='green', lw=1.5),
        ha='center', va='bottom',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="k", lw=1, alpha=0.7)
    )
    
    # 标记坐标点
    plt.text(x_wall_at_min, y_at_min - 0.5, f'({x_wall_at_min:.2f}, {y_at_min:.2f})', ha='right', fontsize=9)
    plt.text(x_car_at_min, y_at_min - 0.5, f'({x_car_at_min:.2f}, {y_at_min:.2f})', ha='left', fontsize=9)

    # 设置图表属性
    plt.title(f'Visualization for y_offset: {wall_params["center"][1]}, wall_angle: {wall_params["theta_deg"]}°', fontsize=16)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.legend()
    plt.axis('equal') # 保持x,y轴比例一致，使角度看起来正确
    plt.grid(True)
    #plt.show()

    # 直接保存图到文件夹
    if save_path:
        plt.savefig(save_path)
    plt.close()

# 辅助函数，用于在绘图时调用
def get_x_car_left(y: float, car_params: Dict[str, Any]) -> float:
    x2, y2 = car_params["center"]
    x_front = x2
    l1 = car_params["l1"]
    alpha_deg = car_params["alpha_deg"]
    
    y_car_top_corner = y2 + l1 / 2
    y_car_bottom_corner = y2 - l1 / 2
    
    if alpha_deg >= 90 or alpha_deg <= 0:
        tan_alpha = float('inf')
    else:
        tan_alpha = math.tan(math.radians(alpha_deg))

    if y_car_bottom_corner <= y <= y_car_top_corner:
        return x_front
    elif y > y_car_top_corner:
        return x_front + (y - y_car_top_corner) * tan_alpha
    else:
        return x_front - (y - y_car_bottom_corner) * tan_alpha

# ----------------------------------------------------------------------------
# 主程序: 定义和运行测试用例
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # 墙车几何参数
    car_l_val = 2.0
    car_l2_val = 0.8
    car_l1_val = car_l_val - car_l2_val
    wall_L_val = 4 * car_l_val
    wall_W_val = car_l_val
    alpha_deg_val = 50 

    wall_x_origin = -0.81 # 墙角度为0时的x坐标，此时墙右侧恰与车前端接近重叠，单位为米
    wall_y_origin = 0.0 # 墙角度为0时的y坐标，此时全宽正碰，单位为米
    overlap_y_list = []

    # 改变overlap_y（-1到1）, wall_theta_deg（-60°到60°）的值来测试不同情况
    for overlap_y in ([-0.995, -0.99] + list(np.arange(-0.9, 1.0, 0.1)) + [0.99, 0.995]):
        for wall_theta_deg in [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]:
            # 打印这俩个参数的值
            print(f"Testing with overlap_y = {overlap_y:.2f}, wall_theta_deg = {wall_theta_deg:.2f}°")
            y_offset = calculate_y_offset(
                wall_L=wall_L_val,
                wall_W=wall_W_val,
                wall_theta_deg=wall_theta_deg,
                car_l=car_l_val,
                overlap_y=overlap_y
            )
            # --- 定义测试用例 ---
            test_cases = { # -0.81m,car_center_x
                "test": {
                    "wall": {"center": (0, y_offset), "L": wall_L_val, "W": wall_W_val, "theta_deg": wall_theta_deg},
                    "car": {"center": (wall_W_val/2, 0), "l": car_l_val, "l1": car_l1_val, "alpha_deg": alpha_deg_val}
                }
            }


            run_and_visualize_case("test", test_cases["test"], './offset_cal_test/test_case_ {:.3f} ; {:.0f} .png'.format(overlap_y, wall_theta_deg))
# %%
