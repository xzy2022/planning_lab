import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# --- 1. 路径设置 (确保能导入 src) ---
# 假设此文件放在 tests/vehicles/ 目录下，向上回溯 3 层找到项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.vehicles.ackermann import AckermannVehicle
from src.vehicles.config import AckermannConfig
from src.vehicles.base import State

def visualize_motion_model():
    # --- 2. 初始化车辆 ---
    config = AckermannConfig(
        wheelbase=2.5,       # 轴距 [m]
        max_steer_deg=35.0   # 最大转角 [deg]
    )
    vehicle = AckermannVehicle(config)
    
    # 初始状态 (在原点，朝向 0 度)
    state = State(x=0.0, y=0.0, theta_rad=0.0)
    
    # --- 3. 仿真参数 ---
    dt = 0.1                 # 时间步长 [s]
    total_time = 15.0        # 总时长 [s]
    steps = int(total_time / dt)
    
    # 记录轨迹用于绘图
    trajectory = [state]
    
    print(f"开始仿真: 总时长 {total_time}s, 步长 {dt}s")

    # --- 4. 仿真循环 (核心) ---
    for i in range(steps):
        t = i * dt
        
        # A. 定义控制输入 (速度 v, 前轮转角 steering)
        velocity = 3.0  # [m/s] 恒定速度
        
        # B. 设定转向逻辑 (S型弯测试)
        if t < 5.0:
            # 第一阶段：左转 (15度)
            steer_input = math.radians(15.0) 
        elif t < 10.0:
            # 第二阶段：直行
            steer_input = 0.0
        else:
            # 第三阶段：右转 (20度)
            steer_input = math.radians(-20.0) 
            
        control = (velocity, steer_input)
        
        # C. 调用运动学模型更新状态
        # kinematic_propagate(current_state, control, dt) -> next_state
        state = vehicle.kinematic_propagate(state, control, dt)
        
        trajectory.append(state)
        
    # --- 5. 可视化 ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # A. 画后轴中心轨迹 (蓝色虚线)
    xs = [s.x for s in trajectory]
    ys = [s.y for s in trajectory]
    ax.plot(xs, ys, 'b--', linewidth=1.5, label='Rear Axle Path')
    
    # B. 画车身姿态 (每隔一定时间画一个快照)
    snapshot_interval = 15 # 每 1.5秒画一次 (15 * 0.1s)
    
    for i, s in enumerate(trajectory):
        if i % snapshot_interval == 0 or i == len(trajectory) - 1:
            # 获取用于显示的车身多边形 (世界坐标系)
            poly_points = vehicle.get_visualization_polygon(s)
            
            # 画车身矩形
            poly = Polygon(poly_points, closed=True, 
                           facecolor='orange', alpha=0.4, edgecolor='k', linewidth=0.5)
            ax.add_patch(poly)
            
            # 画车头朝向箭头 (红色)
            arrow_len = 1.5
            ax.arrow(s.x, s.y, 
                     arrow_len * math.cos(s.theta_rad), 
                     arrow_len * math.sin(s.theta_rad),
                     head_width=0.3, color='darkred', zorder=5)

    # C. 图表修饰
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_title("Ackermann Vehicle Kinematic Simulation\n(Left Turn -> Straight -> Right Turn)")
    ax.set_xlabel("X Position [m]")
    ax.set_ylabel("Y Position [m]")
    
    # 添加信息框
    info_text = (
        f"Sim Time: {total_time}s\n"
        f"Velocity: 3.0 m/s\n"
        f"Wheelbase: {config.wheelbase}m"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.legend()
    plt.tight_layout()
    plt.show()
    print("可视化完成。")

if __name__ == "__main__":
    visualize_motion_model()