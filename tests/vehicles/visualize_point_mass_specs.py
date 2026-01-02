import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

# --- 1. 路径黑魔法 (确保能导入 src) ---
# 假设此文件放在 tests/vehicles/ 目录下，向上回溯 3 层找到 src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.vehicles.point_mass import PointMassVehicle
from src.vehicles.config import PointMassConfig
from src.vehicles.base import State

def draw_dimension_line(ax, start, end, text, offset=0, color='k', vertical=False):
    """
    辅助函数：画尺寸标注线 (支持水平和垂直)
    """
    x1, y1 = start
    x2, y2 = end
    
    if vertical:
        # 垂直标注 (偏移在 x 方向)
        x_line = x1 + offset
        # 1. 界限短线
        ax.plot([x1, x_line], [y1, y1], color=color, linestyle='-', linewidth=0.5, alpha=0.5)
        ax.plot([x2, x_line], [y2, y2], color=color, linestyle='-', linewidth=0.5, alpha=0.5)
        # 2. 箭头线
        ax.annotate('', xy=(x_line, y1), xytext=(x_line, y2),
                    arrowprops=dict(arrowstyle='<->', color=color))
        # 3. 文字
        ax.text(x_line + 0.1, (y1 + y2) / 2.0, text, ha='left', va='center', fontsize=9, color=color, fontweight='bold')
    else:
        # 水平标注 (偏移在 y 方向)
        y_line = y1 + offset
        # 1. 界限短线
        ax.plot([x1, x1], [y1, y_line], color=color, linestyle='-', linewidth=0.5, alpha=0.5)
        ax.plot([x2, x2], [y2, y_line], color=color, linestyle='-', linewidth=0.5, alpha=0.5)
        # 2. 箭头线
        ax.annotate('', xy=(x1, y_line), xytext=(x2, y_line),
                    arrowprops=dict(arrowstyle='<->', color=color))
        # 3. 文字
        ax.text((x1 + x2) / 2.0, y_line + 0.1, text, ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

def visualize_specs():
    # --- 1. 初始化配置 ---
    # 质点模型通常是对称的矩形或正方形
    config = PointMassConfig(
        width=1.5,       # 车宽
        length=2.0,      # 车长
        safe_margin=0.0  # 设为0以便观察精确几何边界
    )
    
    vehicle = PointMassVehicle(config)

    # --- 2. 设定状态 ---
    # 放置在原点，无旋转，方便观察对齐情况
    state = State(x=0.0, y=0.0, theta=30.0)

    # --- 3. 获取几何数据 ---
    vis_poly = vehicle.get_visualization_polygon(state)
    bx, by, b_radius = vehicle.get_bounding_circle(state)

    # --- 4. 绘图 ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # A. 画外接圆 (Bounding Circle)
    # 对于 PointMass，圆心通常就在 state.x, state.y
    bounding_circle = Circle((bx, by), b_radius, 
                             fill=False, linestyle='--', color='magenta', linewidth=1.5,
                             label='Bounding Circle')
    ax.add_patch(bounding_circle)
    
    # B. 画圆心/车辆中心
    ax.plot(state.x, state.y, 'm+', markersize=12, markeredgewidth=2, label='Center (Geometry Center)')

    # C. 画车身 (Visual Body)
    vis_patch = Polygon(vis_poly, closed=True, color='lightgreen', alpha=0.5, edgecolor='green', linewidth=1, label='Vehicle Body')
    ax.add_patch(vis_patch)

    # --- 5. 核心：尺寸标注 (Dimensions) ---
    # 计算矩形角点坐标 (基于 config 和 原点)
    half_len = config.length / 2.0
    half_width = config.width / 2.0
    
    # 关键点
    left_x = -half_len
    right_x = half_len
    bottom_y = -half_width
    top_y = half_width
    
    # 标注 1: Length (车长) - 画在下方
    draw_dimension_line(ax, (left_x, bottom_y), (right_x, bottom_y), 
                        f'Length\n{config.length}m', offset=-0.5, color='blue')

    # 标注 2: Width (车宽) - 画在右侧 (使用垂直模式)
    draw_dimension_line(ax, (right_x, bottom_y), (right_x, top_y), 
                        f'Width\n{config.width}m', offset=0.5, color='darkgreen', vertical=True)

    # --- 6. 辅助虚线 ---
    # 画出中心十字线
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

    # --- 7. 设置显示 ---
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_title("PointMass Vehicle Geometry Specification", fontsize=14)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    
    # 动态调整视野范围
    limit = max(config.length, config.width) + 1.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_specs()