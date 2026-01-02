# tests/vehicles/test_vehicle_viz.py
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

# --- 路径黑魔法 (确保能导入 src) ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.vehicles.ackermann import AckermannVehicle
from src.vehicles.config import AckermannConfig
from src.vehicles.base import State

def visualize_vehicle_layers():
    # 1. 初始化配置和车辆
    config = AckermannConfig()
    # 故意调大安全余量，方便观察外接圆与车身的间隙
    config.safe_margin = 0.15 
    
    vehicle = AckermannVehicle(config)

    # 2. 设定测试状态
    # 车头朝向 30 度
    state = State(x=2.0, y=2.0, theta_rad=math.radians(30))

    # 3. 获取各类几何数据
    
    # [修改点 A] 使用 get_bounding_circle 替代 clearance_radius
    # 返回的是：圆心X, 圆心Y, 半径
    bx, by, b_radius = vehicle.get_bounding_circle(state)
    
    # B. 多圆检测
    cxs, cys, circle_r = vehicle.get_collision_circles(state)
    
    # C. 碰撞多边形
    col_poly = vehicle.get_collision_polygon(state)
    
    # D. 可视化外观
    vis_poly = vehicle.get_visualization_polygon(state)

    # 4. 开始绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # --- 层级 1: 最小外接圆 (Bounding Circle) ---
    # [修改点 B] 圆心不再是 state.x, state.y，而是计算出的 bx, by
    bounding_circle = Circle((bx, by), b_radius, 
                             fill=False, linestyle='--', color='magenta', linewidth=1.5,
                             label='Bounding Circle (AABB Center)')
    ax.add_patch(bounding_circle)
    # 画出外接圆的圆心 (品红色点)，以示区别
    ax.plot(bx, by, 'm+', markersize=10, markeredgewidth=2, label='Bounding Center')

    # --- 层级 2: 碰撞多圆 (Collision Circles) ---
    for i, (cx, cy) in enumerate(zip(cxs, cys)):
        label = 'Collision Circles' if i == 0 else None
        c = Circle((cx, cy), circle_r, color='green', alpha=0.3, label=label)
        ax.add_patch(c)
        ax.plot(cx, cy, 'g.', markersize=2)

    # --- 层级 3: 可视化车身 (Visual Body) ---
    vis_patch = Polygon(vis_poly, closed=True, color='blue', alpha=0.5, label='Visual Body')
    ax.add_patch(vis_patch)

    # --- 层级 4: 碰撞多边形 (Collision Polygon) ---
    col_patch = Polygon(col_poly, closed=True, fill=False, edgecolor='red', linewidth=2, linestyle='-', label='Collision Polygon')
    ax.add_patch(col_patch)

    # --- 辅助信息 ---
    # 画出车辆原点 (后轴中心)
    ax.plot(state.x, state.y, 'ko', label='Rear Axle (Origin)')
    
    # 画出车头朝向箭头
    arrow_len = 1.0
    ax.arrow(state.x, state.y, arrow_len*math.cos(state.theta_rad), arrow_len*math.sin(state.theta_rad), 
             head_width=0.2, head_length=0.3, fc='k', ec='k')

    # 5. 设置图表属性
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'Ackermann Vehicle Geometry Check\n(Theta = {math.degrees(state.theta_rad):.1f} deg)')
    
    # 把图例放外面一点，防止遮挡
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.autoscale_view()
    plt.tight_layout() # 自动调整布局防遮挡
    plt.show()

if __name__ == "__main__":
    visualize_vehicle_layers()