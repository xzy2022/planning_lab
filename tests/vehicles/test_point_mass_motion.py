import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

# --- 1. 路径设置 (确保能导入 src) ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.vehicles.point_mass import PointMassVehicle
from src.vehicles.config import PointMassConfig
from src.vehicles.base import State

def visualize_point_mass_motion():
    # --- 2. 初始化车辆 ---
    # 质点模型通常被视为一个正方形或圆形
    config = PointMassConfig(
        width=1.0,       # 车宽 [m]
        length=2.0,      # 车长 [m]
    )
    vehicle = PointMassVehicle(config)
    
    # 初始状态 (在原点)
    # 注意：对于质点模型，theta 通常保持不变，或者代表"车头"的固定朝向
    state = State(x=0.0, y=0.0, theta_rad=1.0)
    
    # --- 3. 仿真参数 ---
    dt = 0.1
    # 走一个正方形：右 -> 上 -> 左 -> 下
    phase_duration = 3.0 # 每个方向走3秒
    total_time = phase_duration * 4
    steps = int(total_time / dt)
    
    trajectory = [state]
    print(f"开始仿真: 质点全向移动测试 (走正方形), 总时长 {total_time}s")

    # --- 4. 仿真循环 ---
    for i in range(steps):
        t = i * dt
        
        # 定义控制输入 (vx, vy) - 全向移动不需要转向
        speed = 2.0
        
        if t < phase_duration:
            # 阶段1: 向右 (vx > 0)
            control = (speed, 0.0)
        elif t < phase_duration * 2:
            # 阶段2: 向上 (vy > 0) - 注意车身无需旋转
            control = (0.0, speed)
        elif t < phase_duration * 3:
            # 阶段3: 向左 (vx < 0)
            control = (-speed, 0.0)
        else:
            # 阶段4: 向下 (vy < 0)
            control = (0.0, -speed)
            
        # 物理推演
        state = vehicle.kinematic_propagate(state, control, dt)
        trajectory.append(state)

    # --- 5. 可视化 ---
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # A. 画轨迹 (蓝色虚线)
    xs = [s.x for s in trajectory]
    ys = [s.y for s in trajectory]
    ax.plot(xs, ys, 'b--', linewidth=1.5, label='Center Trajectory')
    
    # B. 画车身快照
    # 因为质点模型不旋转，车身应该始终保持正方形姿态
    snapshot_interval = 10 # 每1秒画一次
    
    for i, s in enumerate(trajectory):
        if i % snapshot_interval == 0 or i == len(trajectory) - 1:
            # 获取外观多边形
            poly_points = vehicle.get_visualization_polygon(s)
            
            # 颜色随时间变化，方便看清顺序 (浅黄 -> 深橙)
            alpha_time = 0.3 + 0.6 * (i / steps)
            
            poly = Polygon(poly_points, closed=True, 
                           facecolor='orange', alpha=0.5, edgecolor='brown', linewidth=1)
            ax.add_patch(poly)
            
            # 画一个十字标表示中心
            ax.plot(s.x, s.y, 'k+', markersize=5)

    # C. 图表修饰
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    limit = 8.0
    ax.set_xlim(-2, limit)
    ax.set_ylim(-2, limit)
    
    ax.set_title("PointMass Holonomic Motion (Square Path)\nNotice: Orientation (Theta) does not change")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    
    # 添加说明
    info_text = (
        f"Control Input: (vx, vy)\n"
        f"Width/Length: {config.width}m\n"
        "Type: Holonomic (Omnidirectional)"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    print("可视化完成。")

if __name__ == "__main__":
    visualize_point_mass_motion()