# tests/map_test/test_map_generation.py
import sys
import os
import matplotlib.pyplot as plt

# --- 路径挂载 (确保能导入 src) ---
# 获取当前文件所在目录的上上级目录 (即项目根目录)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.map.grid_map import GridMap

def test_map_visualization():
    print("开始地图生成测试...")
    
    # 1. 初始化参数
    width = 100
    height = 100
    resolution = 0.1
    
    # 2. 创建地图实例
    grid_map = GridMap(width, height, resolution)
    
    # 3. 步骤一：随机生成
    print("Step 1: 生成随机障碍物 (密度 0.05)")
    grid_map.generate_random_obstacles(density=0.05, seed=42)
    
    # 保存一下原始数据用于对比绘图
    original_data = grid_map.data.copy()
    
    # 4. 步骤二：膨胀
    print("Step 2: 膨胀障碍物 (半径 2 格)")
    grid_map.inflate_obstacles(radius_grids=2)
    inflated_data = grid_map.data.copy()

    # 5. 可视化对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 图1：原始随机点
    ax1.imshow(original_data, cmap='Greys', origin='lower')
    ax1.set_title("Step 1: Random Obstacles")
    ax1.set_xlabel("X (grids)")
    ax1.set_ylabel("Y (grids)")

    # 图2：膨胀后的迷宫/洞穴
    ax2.imshow(inflated_data, cmap='Greys', origin='lower')
    ax2.set_title("Step 2: Inflated (Maze-like)")
    ax2.set_xlabel("X (grids)")
    
    plt.suptitle(f"Map Generation Test ({width}x{height})")
    plt.tight_layout()
    plt.show()
    print("测试完成。")

if __name__ == "__main__":
    test_map_visualization()