import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch

# --- 1. 路径黑魔法 (确保能导入 src) ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.vehicles.ackermann import AckermannVehicle
from src.vehicles.config import AckermannConfig
from src.vehicles.base import State

def draw_dimension_line(ax, start, end, text, offset_y=0, color='k'):
    """
    辅助函数：画类似工程制图的尺寸标注线
    :param start: (x, y) 起点
    :param end: (x, y) 终点
    :param text: 标注文字
    :param offset_y: y轴方向的偏移量（避免重叠）
    """
    x1, y1 = start
    x2, y2 = end
    y_line = y1 + offset_y
    
    # 1. 画两端的短竖线 (界限)
    ax.plot([x1, x1], [y1, y_line], color=color, linestyle='-', linewidth=0.5, alpha=0.5)
    ax.plot([x2, x2], [y2, y_line], color=color, linestyle='-', linewidth=0.5, alpha=0.5)
    
    # 2. 画带箭头的横线
    ax.annotate('', xy=(x1, y_line), xytext=(x2, y_line),
                arrowprops=dict(arrowstyle='<->', color=color))
    
    # 3. 标文字 (居中)
    mid_x = (x1 + x2) / 2.0
    ax.text(mid_x, y_line + 0.1, text, ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

def visualize_specs():
    # --- 1. 初始化配置 (使用非对称参数以区分前后) ---
    config = AckermannConfig(
        wheelbase=3.0,       # 轴距加长
        front_hang=1.0,      # 前悬
        rear_hang=0.8,       # 后悬 (比前悬短)
        width=1.8,           # 车宽
        safe_margin = 0.0    # 暂时设为0，以便精确观察几何贴合度
    )

    
    # 重新触发计算
    vehicle = AckermannVehicle(config)

    # --- 2. 设定状态 (水平放置，方便标注) ---
    state = State(x=0.0, y=0.0, theta=0.0)

    # --- 3. 获取几何数据 ---
    vis_poly = vehicle.get_visualization_polygon(state)
    bx, by, b_radius = vehicle.get_bounding_circle(state)

    # --- 4. 绘图 ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # A. 画外接圆 (Bounding Circle)
    bounding_circle = Circle((bx, by), b_radius, 
                             fill=False, linestyle='--', color='magenta', linewidth=1.5,
                             label='Bounding Circle')
    ax.add_patch(bounding_circle)
    
    # B. 画外接圆圆心 (Bounding Center)
    ax.plot(bx, by, 'm+', markersize=12, markeredgewidth=2, label='Bounding Center (Geometry Center)')

    # C. 画车身 (Visual Body)
    vis_patch = Polygon(vis_poly, closed=True, color='lightblue', alpha=0.5, edgecolor='blue', linewidth=1, label='Vehicle Body')
    ax.add_patch(vis_patch)

    # D. 画关键点：后轴中心 (原点) & 前轴中心
    ax.plot(state.x, state.y, 'ko', markersize=8, label='Rear Axle Center (Origin / x=0)')
    
    front_axle_x = state.x + config.wheelbase
    ax.plot(front_axle_x, state.y, 'ko', fillstyle='none', markersize=8, label='Front Axle Center')

    # --- 5. 核心：尺寸标注 (Dimensions) ---
    # 关键 x 坐标
    rear_end_x = -config.rear_hang
    rear_axle_x = 0
    front_axle_x = config.wheelbase
    front_end_x = config.wheelbase + config.front_hang
    
    y_bottom = -config.width / 2.0
    
    # 标注 1: Rear Hang (后悬)
    draw_dimension_line(ax, (rear_end_x, y_bottom), (rear_axle_x, y_bottom), 
                        f'Rear Hang\n{config.rear_hang}m', offset_y=-0.5, color='darkgreen')

    # 标注 2: Wheelbase (轴距)
    draw_dimension_line(ax, (rear_axle_x, y_bottom), (front_axle_x, y_bottom), 
                        f'Wheelbase\n{config.wheelbase}m', offset_y=-0.5, color='blue')

    # 标注 3: Front Hang (前悬)
    draw_dimension_line(ax, (front_axle_x, y_bottom), (front_end_x, y_bottom), 
                        f'Front Hang\n{config.front_hang}m', offset_y=-0.5, color='darkgreen')
    
    # 标注 4: Width (车宽) - 画在右侧
    x_right = front_end_x + 0.5
    draw_dimension_line(ax, (x_right, -config.width/2), (x_right, config.width/2), 
                        f'Width {config.width}m', offset_y=0) # 这里的实现需要简单修改draw支持垂直，或者手动画
    # 手动补一个垂直标注
    ax.annotate('', xy=(front_end_x, -config.width/2), xytext=(front_end_x, config.width/2),
                arrowprops=dict(arrowstyle='<->', color='purple'))
    ax.text(front_end_x + 0.1, 0, f'Width\n{config.width}m', va='center', color='purple')

    # --- 6. 辅助虚线 ---
    # 画出前轴、后轴、车头、车尾的垂线
    for x in [rear_end_x, rear_axle_x, front_axle_x, front_end_x]:
        ax.axvline(x, color='gray', linestyle=':', alpha=0.5)

    # --- 7. 设置显示 ---
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_title("Ackermann Vehicle Geometry Specification", fontsize=14)
    ax.set_xlabel("Longitudinal (x) [m]")
    ax.set_ylabel("Lateral (y) [m]")
    
    # 增加图例
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_specs()