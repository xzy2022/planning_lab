import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

# --- 1. 路径黑魔法 (确保能导入 src) ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.vehicles.ackermann import AckermannVehicle, AckermannConfig
from src.vehicles.base import State
from src.map.grid_map import GridMap
from src.collision.checker import CollisionChecker
from src.collision.config import CollisionConfig, CollisionMethod

def setup_environment():
    """准备地图和车辆"""
    # 1. 创建一个小地图 (10m x 10m)
    grid_map = GridMap(width=100, height=100, resolution=0.1)
    
    # 2. 在地图中间放置一个矩形障碍物 (模拟墙壁或柱子)
    # 障碍物位置: x=[5.0, 5.5], y=[4.0, 6.0]
    for y in range(40, 61):
        for x in range(50, 56):
            grid_map.data[y, x] = 1
            
    # 3. 初始化车辆
    vehicle_config = AckermannConfig()
    vehicle = AckermannVehicle(vehicle_config)
    
    return grid_map, vehicle

def draw_map(ax, grid_map):
    """画底图障碍物"""
    # 使用 imshow 显示栅格
    # origin='lower' 确保 (0,0) 在左下角
    # cmap='Greys' 让 0 是白色，1 是黑色
    ax.imshow(grid_map.data, origin='lower', cmap='Greys', 
              extent=[0, grid_map.width * grid_map.resolution, 
                      0, grid_map.height * grid_map.resolution],
              alpha=0.5)

def visualize_collision_test():
    grid_map, vehicle = setup_environment()
    
    # --- 定义测试场景 (State) ---
    test_states = [
        # 1. 安全位置 (远离障碍)
        State(x=2.0, y=5.0, theta_rad=math.radians(90)),
        
        # 2. 临界状态 A: 外接圆碰到了，但车身没碰到 (Broad Fail, Narrow Pass)
        # 障碍物在 x=5.0。车宽2.0，外接圆半径约2.0。
        # 车心在 3.5，车头在 3.5+3.4(全长) 不太对。
        # 调整：障碍物 x=5.0。车尾 x=2.5, 车头 x=5.0+。
        # 让车身紧贴障碍物左侧
        State(x=3.2, y=4.95, theta_rad=math.radians(90)), 
        
        # 3. 碰撞状态 B: 车头插入障碍物
        State(x=3.9, y=5.0, theta_rad=math.radians(90)),
        
        # 4. 复杂状态: 斜着停在障碍物旁边 (测试多圆/SAT的贴合度)
        State(x=3.5, y=3.5, theta_rad=math.radians(45))
    ]

    # --- 定义要测试的检测模式 ---
    methods = [
        (CollisionMethod.CIRCLE_ONLY, "Broad Phase Only (Circle)"),
        (CollisionMethod.MULTI_CIRCLE, "Narrow Phase (Multi-Circle)"),
        (CollisionMethod.POLYGON, "Narrow Phase (Polygon SAT)")
    ]

    # --- 开始绘图 ---
    fig, axes = plt.subplots(len(test_states), len(methods), figsize=(15, 12))
    
    # 如果只有一行或一列，确保 axes 是二维数组
    if len(test_states) == 1: axes = np.array([axes])
    if len(methods) == 1: axes = axes[:, np.newaxis]

    for row, state in enumerate(test_states):
        for col, (method, method_name) in enumerate(methods):
            ax = axes[row, col]
            
            # 1. 绘制地图背景
            draw_map(ax, grid_map)
            
            # 2. 初始化检测器并检测
            config = CollisionConfig(method=method)
            checker = CollisionChecker(config)
            is_collision = checker.check(vehicle, state, grid_map)
            
            # 3. 结果标题颜色 (红=撞, 绿=安全)
            color = 'red' if is_collision else 'green'
            result_text = "COLLISION" if is_collision else "SAFE"
            ax.set_title(f"{method_name}\nResult: {result_text}", color=color, fontweight='bold', fontsize=10)
            
            # 4. 可视化几何体 (画出算法"眼中"看到的形状)
            
            # 总是画出车身轮廓作为参考 (蓝色虚线)
            vis_poly = vehicle.get_visualization_polygon(state)
            ax.add_patch(Polygon(vis_poly, closed=True, fill=False, edgecolor='blue', linestyle='--', alpha=0.5))
            
            # A. 针对 CIRCLE 模式：画外接圆
            if method == CollisionMethod.CIRCLE_ONLY:
                bx, by, br = vehicle.get_bounding_circle(state)
                # 如果检测认为撞了，用红色填充，否则绿色空心
                fill_color = 'red' if is_collision else 'green'
                alpha = 0.2 if is_collision else 0.0
                ax.add_patch(Circle((bx, by), br, color=fill_color, alpha=alpha, fill=True))
                ax.add_patch(Circle((bx, by), br, edgecolor=color, fill=False, linewidth=2))
                ax.plot(bx, by, 'x', color=color) # 圆心

            # B. 针对 MULTI_CIRCLE 模式：画一串小圆
            elif method == CollisionMethod.MULTI_CIRCLE:
                cxs, cys, r = vehicle.get_collision_circles(state)
                for cx, cy in zip(cxs, cys):
                    # 画圆
                    c = Circle((cx, cy), r, color=color, alpha=0.2)
                    ax.add_patch(c)
                    ax.plot(cx, cy, '.', color=color, markersize=2)
                
                # 同时画出外接圆(灰色虚线)表示 Broad Phase 范围
                bx, by, br = vehicle.get_bounding_circle(state)
                ax.add_patch(Circle((bx, by), br, edgecolor='gray', linestyle=':', fill=False))

            # C. 针对 POLYGON 模式：画碰撞多边形
            elif method == CollisionMethod.POLYGON:
                poly_points = vehicle.get_collision_polygon(state)
                poly_patch = Polygon(poly_points, closed=True, facecolor=color, alpha=0.3, edgecolor=color)
                ax.add_patch(poly_patch)
                
                # Broad Phase 范围
                bx, by, br = vehicle.get_bounding_circle(state)
                ax.add_patch(Circle((bx, by), br, edgecolor='gray', linestyle=':', fill=False))

            # 设置视野
            ax.set_aspect('equal')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.grid(True, linestyle=':', alpha=0.3)
            
            # 仅在左侧标注行信息
            if col == 0:
                ax.set_ylabel(f"State {row+1}\nx={state.x:.1f}, y={state.y:.1f}\ntheta={math.degrees(state.theta_rad):.0f}°")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_collision_test()