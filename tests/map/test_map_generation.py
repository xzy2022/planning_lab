# tests/map/test_map_generation.py
import sys
import os
import matplotlib.pyplot as plt

# 路径黑魔法
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.ackermann import AckermannVehicle
from src.vehicles.config import AckermannConfig
from src.types import State

def test_feasible_map_generation():
    # 1. 准备组件
    # [修正] 扩大地图尺寸以容纳 (90, 90) 的目标点
    # 1000 grids * 0.1m/grid = 100m x 100m
    grid_map = GridMap(200, 200, resolution=0.5)
    
    config = AckermannConfig(
        wheelbase=2.5, 
        max_steer_deg=34.4
    )
    vehicle = AckermannVehicle(config)
    
    start = State(5.0, 5.0, 0.0)
    goal = State(90.0, 90.0, 0.0)
    
    # 2. 调用生成器
    # 计算物理尺寸用于打印和可视化
    phys_width = grid_map.width * grid_map.resolution
    phys_height = grid_map.height * grid_map.resolution

    print(f"Map Physical Size: {phys_width:.1f}m x {phys_height:.1f}m")
    print("生成包含可行路径的地图中...")
    
    MapGenerator.generate_feasible_map(
        grid_map, vehicle, start, goal, obstacle_density=0.1
    )
    
    # 3. 可视化
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # [关键修改] 使用 extent 参数设置物理坐标范围
    # 格式: [left, right, bottom, top]
    # 对应: [0, 物理宽, 0, 物理高]
    map_extent = [0, phys_width, 0, phys_height]

    # A. 画地图 
    # extent=map_extent 让 Matplotlib 自动把像素拉伸到物理坐标系
    ax.imshow(grid_map.data, cmap='Greys', origin='lower', extent=map_extent)
    
    # B. 画起点和终点
    ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
    ax.plot(goal.x, goal.y, 'rx', markersize=10, markeredgewidth=2, label='Goal')
    
    ax.set_title(f"Feasible Map Generation (Physical Coordinates)\n"
                 f"Density: 0.6, Resolution: {grid_map.resolution}m")
    
    ax.set_xlabel("X Position [m]")
    ax.set_ylabel("Y Position [m]")
    
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.3) # 加个网格更好看
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_feasible_map_generation()