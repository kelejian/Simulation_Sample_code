# %%
import math, os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from typing import Tuple, Dict, Any

# ----------------------------------------------------------------------------
# 几何辅助函数 (用于获取完整顶点，支持可视化)
# ----------------------------------------------------------------------------
def get_wall_vertices(center: Tuple[float, float], W: float, L: float, theta_deg: float) -> np.ndarray:
    """
    计算旋转后墙体的四个顶点坐标。
    
    Args:
        center: 墙的中心坐标 (x, y)。
        W: 墙的宽度。
        L: 墙的长度。
        theta_deg: 墙的逆时针旋转角度（度）。

    Returns:
        np.ndarray: 4x2的数组，包含墙的四个顶点坐标 [左上, 右上, 右下, 左下]。
    """
    half_W, half_L = W / 2, L / 2
    base_vertices = np.array([
        [-half_W,  half_L],  # 左上
        [ half_W,  half_L],  # 右上
        [ half_W, -half_L],  # 右下
        [-half_W, -half_L]   # 左下
    ])

    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    rotation_matrix = np.array([[c, -s], [s, c]])
    rotated_vertices = base_vertices @ rotation_matrix.T

    return rotated_vertices + np.array(center)


def get_car_vertices(center: Tuple[float, float], l: float, l1: float, alpha_deg: float) -> np.ndarray:
    """
    计算车头轮廓的顶点坐标（带倒角的六边形前端）。
    
    Args:
        center: 车的中心坐标 (x, y)，对应车头平直部分的x坐标。
        l: 车的总宽度。
        l1: 车头平直部分的长度。
        alpha_deg: 倒角与y轴的夹角（度）。

    Returns:
        np.ndarray: 6x2的数组，包含车前端的六个顶点坐标。
    """
    x_c, y_c = center
    half_l, half_l1 = l / 2, l1 / 2
    l2 = l - l1
    alpha_rad = np.deg2rad(alpha_deg)
    
    # 倒角在x方向的偏移
    dx_chamfer = (l2 / 2) * np.tan(alpha_rad)
    x_front = x_c  # 车头平直部分的x坐标
    
    # 定义车头轮廓的6个顶点
    car_vertices = np.array([
        [x_front, y_c + half_l1],              # 平直部分上端点
        [x_front + dx_chamfer, y_c + half_l],  # 上倒角外端点
        [x_c + 2, y_c + half_l],               # 车身右上角 (示意)
        [x_c + 2, y_c - half_l],               # 车身右下角 (示意)
        [x_front + dx_chamfer, y_c - half_l],  # 下倒角外端点
        [x_front, y_c - half_l1],              # 平直部分下端点
    ])

    return car_vertices


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
# 可视化函数用以验证 (使用多边形绘制)
# ----------------------------------------------------------------------------
def run_and_visualize_case(case_name: str, params: Dict[str, Any], save_path: str = None) -> None:
    """运行单个测试用例并生成可视化图表（使用多边形绘制墙和车）"""
    
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
    
    # 1. 获取墙的完整顶点 (平移前和平移后)
    wall_vertices = get_wall_vertices(
        center=wall_params["center"],
        W=wall_params["W"],
        L=wall_params["L"],
        theta_deg=wall_params["theta_deg"]
    )
    shifted_wall_vertices = wall_vertices + np.array([x_offset, 0])
    
    # 2. 获取车的顶点
    car_vertices = get_car_vertices(
        center=car_params["center"],
        l=car_params["l"],
        l1=car_params["l1"],
        alpha_deg=car_params["alpha_deg"]
    )

    # --- 开始绘图 ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制车 (蓝色多边形)
    ax.add_patch(Polygon(car_vertices, closed=True, facecolor='royalblue', 
                         edgecolor='black', alpha=0.7, label=f'Car (α={car_params["alpha_deg"]}°)'))
    
    # 绘制原始位置的墙 (灰色半透明多边形)
    ax.add_patch(Polygon(wall_vertices, closed=True, facecolor='gray', 
                         alpha=0.4, edgecolor='black', label=f'Wall (before, θ={wall_params["theta_deg"]}°)'))
    
    # 绘制平移后的墙 (红色虚线边框)
    ax.add_patch(Polygon(shifted_wall_vertices, closed=True, facecolor='none', 
                         edgecolor='red', linestyle='--', linewidth=2, 
                         label=f'Wall (after shift Δx={x_offset:.3f})'))
    
    # 标记最小距离点和连接线
    x_wall_at_min, y_at_min = min_gap_point_on_wall
    x_car_at_min = x_wall_at_min + x_offset
    
    ax.plot([x_wall_at_min, x_car_at_min], [y_at_min, y_at_min], 'k--', lw=1)
    ax.scatter([x_wall_at_min, x_car_at_min], [y_at_min, y_at_min], c='black', zorder=5, s=30)

    # 添加箭头和文本标注
    mid_x = (x_wall_at_min + x_car_at_min) / 2
    ax.annotate(
        f'Δx = {x_offset:.3f}',
        xy=(x_wall_at_min, y_at_min),
        xytext=(mid_x, y_at_min + 0.5),
        arrowprops=dict(arrowstyle='<->', color='green', lw=1.5),
        ha='center', va='bottom',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="k", lw=1, alpha=0.7)
    )
    
    # 标记坐标点
    ax.text(x_wall_at_min, y_at_min - 0.3, f'({x_wall_at_min:.2f}, {y_at_min:.2f})', ha='right', fontsize=9)
    ax.text(x_car_at_min, y_at_min - 0.3, f'({x_car_at_min:.2f}, {y_at_min:.2f})', ha='left', fontsize=9)
    
    # 标记中心点
    ax.plot(*wall_params["center"], 'ko', markersize=5)
    ax.plot(*car_params["center"], 'bo', markersize=5)

    # 设置图表属性
    ax.set_title(f'Case: {case_name} | y_offset: {wall_params["center"][1]:.2f}, wall_angle: {wall_params["theta_deg"]}°', fontsize=14)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.legend(loc='upper right')
    ax.axis('equal')
    ax.grid(True, linestyle=':')

    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ----------------------------------------------------------------------------
# 主程序: 定义和运行测试用例
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # 墙车几何参数
    car_l_val = 2.0
    car_l2_val = 0.8
    car_l1_val = car_l_val - car_l2_val
    wall_L_val = 2 * car_l_val
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

            # 运行并可视化测试用例
            save_dir = './offset_cal_test/'
            if os.path.exists(save_dir) == False:
                os.makedirs(save_dir)
            run_and_visualize_case("test", test_cases["test"], './offset_cal_test/test_case_ {:.3f} ; {:.0f} .png'.format(overlap_y, wall_theta_deg))
# %%
