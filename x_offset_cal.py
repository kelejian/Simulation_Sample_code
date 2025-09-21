import numpy as np

def get_wall_right_boundary(center, W, L, theta_deg):
    """
    计算旋转后墙体的顶点，并返回其右边界函数。
    
    Args:
        center (np.array): 墙的中心坐标 [x, y]。
        W (float): 墙的宽度。
        L (float): 墙的长度。
        theta_deg (float): 墙的逆时针旋转角度（度）。

    Returns:
        tuple: (
            np.array: 4x2的数组，包含墙的四个顶点坐标。
            function: 一个函数 x_wall_right(y)，输入y坐标，返回墙右边界对应的x坐标。
        )
    """
    # 1. 定义未旋转时，相对于中心的四个顶点
    half_W, half_L = W / 2, L / 2
    base_vertices = np.array([
        [-half_W,  half_L],  # 左上
        [ half_W,  half_L],  # 右上
        [ half_W, -half_L],  # 右下
        [-half_W, -half_L]   # 左下
    ])

    # 2. 创建旋转矩阵并旋转顶点
    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    rotation_matrix = np.array([[c, -s], [s, c]])
    rotated_vertices = base_vertices @ rotation_matrix.T

    # 3. 平移到世界坐标
    wall_vertices = rotated_vertices + center

    # 4. 找到右边界的三个关键顶点（最右点及其相邻点）
    rightmost_idx = np.argmax(wall_vertices[:, 0])
    # 顶点按顺序连接，所以相邻点是 (idx-1)%4 和 (idx+1)%4
    v_right = wall_vertices[rightmost_idx]
    v_adj1 = wall_vertices[(rightmost_idx - 1 + 4) % 4]
    v_adj2 = wall_vertices[(rightmost_idx + 1) % 4]
    
    # 按y坐标排序，方便插值
    right_boundary_points = sorted([v_right, v_adj1, v_adj2], key=lambda p: p[1])
    
    # 5. 定义并返回右边界函数
    def x_wall_right(y):
        # 使用线性插值来计算任意y处的x坐标
        # np.interp(y, y_coords, x_coords)
        y_coords = [p[1] for p in right_boundary_points]
        x_coords = [p[0] for p in right_boundary_points]
        return np.interp(y, y_coords, x_coords)

    return wall_vertices, x_wall_right


def get_car_left_boundary(center, l, l1, alpha_deg):
    """
    计算车体顶点，并返回其左边界函数。
    
    Args:
        center (np.array): 车的中心坐标 [x, y]。
        l (float): 车的总宽度。
        l1 (float): 车头平直部分的长度。
        alpha_deg (float): 倒角与y轴的夹角（度）。

    Returns:
        tuple: (
            np.array: 6x2的数组，包含车前端的六个顶点坐标（用于绘图）。
            function: 一个函数 x_car_left(y)，输入y坐标，返回车左边界对应的x坐标。
        )
    """
    # 1. 计算几何参数
    x_c, y_c = center
    half_l, half_l1 = l / 2, l1 / 2
    l2 = l - l1
    alpha_rad = np.deg2rad(alpha_deg)
    
    # 倒角在x方向的偏移
    dx_chamfer = (l2 / 2) * np.tan(alpha_rad)
    
    # 假设车的后端x坐标是 x_c + some_length，我们只关心前端
    # 设车头平直部分的x坐标为 x_front
    x_front = x_c - dx_chamfer # 假设O2是原始矩形的中心
    
    # 2. 定义车头轮廓的6个顶点（为了方便绘图）
    car_vertices = np.array([
        [x_front, y_c + half_l1],              # 平直部分上端点
        [x_front + dx_chamfer, y_c + half_l],  # 上倒角外端点
        [x_c + 5, y_c + half_l],              # 随便画一个车身右上角
        [x_c + 5, y_c - half_l],              # 随便画一个车身右下角
        [x_front + dx_chamfer, y_c - half_l],  # 下倒角外端点
        [x_front, y_c - half_l1],              # 平直部分下端点
    ])

    # 3. 定义并返回左边界函数
    y_upper_corner = y_c + half_l1
    y_lower_corner = y_c - half_l1
    
    def x_car_left(y):
        if y >= y_upper_corner: # 上方倒角
            return x_front + (y - y_upper_corner) * np.tan(alpha_rad)
        elif y <= y_lower_corner: # 下方倒角
            return x_front - (y - y_lower_corner) * np.tan(alpha_rad)
        else: # 中间平直部分
            return x_front

    return car_vertices, x_car_left


def calculate_min_horizontal_distance(wall_params, car_params):
    """
    计算墙和车之间的最小水平距离 (Δx)。

    Args:
        wall_params (dict): 墙的参数 {center, W, L, theta_deg}
        car_params (dict): 车的参数 {center, l, l1, alpha_deg}

    Returns:
        float: 最小水平距离 Δx。
    """
    # 1. 获取两个物体的边界函数和顶点
    wall_vertices, x_wall_right = get_wall_right_boundary(**wall_params)
    car_vertices, x_car_left = get_car_left_boundary(**car_params)

    # 2. 确定Y轴投影的重叠区间
    y_wall_min, y_wall_max = wall_vertices[:, 1].min(), wall_vertices[:, 1].max()  # 
    y_car_min, y_car_max = car_params['center'][1] - car_params['l']/2, car_params['center'][1] + car_params['l']/2
    
    y_overlap_start = max(y_wall_min, y_car_min)
    y_overlap_end = min(y_wall_max, y_car_max)

    if y_overlap_start >= y_overlap_end:
        print("警告: Y轴无重叠，无法计算距离。")
        return None

    # 3. 找出所有“关键Y坐标点”
    # 关键点 = 重叠区间的端点 + 边界函数的拐点（如果在重叠区间内）
    critical_ys = {y_overlap_start, y_overlap_end}

    # 墙的拐点 (最右顶点的y坐标)
    rightmost_wall_vertex_y = wall_vertices[np.argmax(wall_vertices[:, 0]), 1]
    if y_overlap_start < rightmost_wall_vertex_y < y_overlap_end:
        critical_ys.add(rightmost_wall_vertex_y)

    # 车的拐点
    y_c, l1 = car_params['center'][1], car_params['l1']
    y_car_corner_upper = y_c + l1 / 2
    y_car_corner_lower = y_c - l1 / 2
    if y_overlap_start < y_car_corner_upper < y_overlap_end:
        critical_ys.add(y_car_corner_upper)
    if y_overlap_start < y_car_corner_lower < y_overlap_end:
        critical_ys.add(y_car_corner_lower)

    # 4. 在所有关键点上计算水平间隙，找到最小值
    min_gap = float('inf')
    for y in sorted(list(critical_ys)):
        gap = x_car_left(y) - x_wall_right(y)
        if gap < min_gap:
            min_gap = gap
            
    return min_gap

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def run_validation_cases(num_cases=20):
    """
    生成、计算并可视化多个测试案例。
    """
    # --- 固定参数 ---
    # 车的参数 
    l = 2.0
    l1 = 0.8  # l1 < l
    alpha_deg = 50
    car_center = np.array([10.0, 0.0]) # 将车固定在右侧
    car_params = {'center': car_center, 'l': l, 'l1': l1, 'alpha_deg': alpha_deg}

    # 墙的参数
    W = 2.0 # W=l
    L = 6.0 # L=3l
    
    # --- 生成案例并可视化 ---
    for i in range(num_cases):
        # 随机生成墙的位置和角度，确保它在车的左边且有y轴重叠
        wall_center_x = np.random.uniform(0, 5)
        wall_center_y = np.random.uniform(-1, 1)
        theta_deg = np.random.uniform(-60, 60)

        wall_center_y = 2.0
        theta_deg = 60
        
        wall_params = {
            'center': np.array([wall_center_x, wall_center_y]),
            'W': W, 'L': L, 'theta_deg': theta_deg
        }

        # 1. 计算最小距离
        delta_x = calculate_min_horizontal_distance(wall_params, car_params)
        
        if delta_x is None or delta_x < 0:
            print(f"案例 {i+1} 无效 (无重叠或初始已碰撞)，跳过。")
            continue

        # 2. 准备绘图数据
        wall_vertices, _ = get_wall_right_boundary(**wall_params)
        car_vertices, _ = get_car_left_boundary(**car_params)
        
        # 计算平移后的墙的顶点
        shifted_wall_vertices = wall_vertices + np.array([delta_x, 0])

        # 3. 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制车
        ax.add_patch(Polygon(car_vertices, closed=True, facecolor='royalblue', edgecolor='black', label='Car'))
        
        # 绘制原始位置的墙
        ax.add_patch(Polygon(wall_vertices, closed=True, facecolor='gray', alpha=0.5, edgecolor='black', label='Original Wall'))
        
        # 绘制平移Δx后的墙
        ax.add_patch(Polygon(shifted_wall_vertices, closed=True, facecolor='none', edgecolor='red', linestyle='--', linewidth=2, label=f'Wall Shifted by Δx={delta_x:.3f}'))

        # 标注中心点
        ax.plot(*car_params['center'], 'ko', label='Car Center O₂')
        ax.plot(*wall_params['center'], 'ko', label='Wall Center O₁')

        # 设置图表
        ax.legend()
        ax.set_title(f'Case {i+1}: θ = {theta_deg:.1f}°, Wall Center = ({wall_center_x:.2f}, {wall_center_y:.2f})')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.grid(True, linestyle=':')
        ax.axis('equal') # 保证x,y轴比例相同，形状不失真
        
        plt.show()


# --- 运行验证 ---
if __name__ == '__main__':
    run_validation_cases(num_cases=1)