# tests/map/test_map_generation_comparison.py
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# --- 路径设置 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.point_mass import PointMassVehicle, PointMassConfig
from src.types import State

def test_map_generation_comparison():
    # --- 1. 全局配置 ---
    # 地图尺寸
    width_indices = 100
    height_indices = 100
    resolution = 0.5  # [米/像素]
    
    phys_width = width_indices * resolution
    phys_height = height_indices * resolution
    
    start = State(2.0, 2.0, 0.0)
    goal = State(48.0, 48.0, 0.0)

    # --- 2. 实验变量设置 ---
    # 行变量：车辆配置
    v_normal_cfg = PointMassConfig(width=1.0, length=1.0, safe_margin=0.1)
    v_big_cfg = PointMassConfig(width=2.0, length=2.0, safe_margin=0.1) # 稍微加大一点以便观察差异
    
    vehicles = [
        ("Normal Vehicle (1.0x1.0)", PointMassVehicle(v_normal_cfg)),
        ("Big Vehicle (2.0x2.0)", PointMassVehicle(v_big_cfg))
    ]
    
    # 列变量：障碍物密度
    densities = [0.05, 0.15, 0.25]

    # --- 3. 初始化画布 (2行 3列) ---
    fig, axes = plt.subplots(len(vehicles), len(densities), figsize=(18, 12))
    
    print(f"Starting generation comparison: {len(vehicles)} vehicles x {len(densities)} densities...")

    # --- 4. 双层循环生成与绘制 ---
    for row_idx, (v_name, vehicle) in enumerate(vehicles):
        for col_idx, density in enumerate(densities):
            ax = axes[row_idx, col_idx]
            
            print(f"Generating -> Row: {row_idx} ({v_name}), Col: {col_idx} (Density {density})")
            
            # A. 每次都要创建一个新的干净地图
            grid_map = GridMap(width_indices, height_indices, resolution=resolution)
            
            # B. 初始化生成器
            # inflation_radius_m: 静态障碍物自身的膨胀（模拟墙厚度），与车身无关
            generator = MapGenerator(
                obstacle_density=density, 
                inflation_radius_m=0.5, 
                num_waypoints=4
            )
            
            # C. 执行生成 (挖空路径)
            # extra_paths=0, dead_ends=3 以保持图像相对整洁
            generator.generate(grid_map, vehicle, start, goal, extra_paths=0, dead_ends=3)
            
            # D. 绘图
            map_extent = [0, phys_width, 0, phys_height]
            
            # 1. 画地图障碍物
            ax.imshow(grid_map.data, cmap='Greys', origin='lower', extent=map_extent)
            
            # 2. 画起点终点位置
            ax.plot(start.x, start.y, 'go', markersize=4)
            ax.plot(goal.x, goal.y, 'rx', markersize=4)
            
            # 3. 画车身轮廓 (验证生成器是否留出了足够空间)
            start_poly = vehicle.get_visualization_polygon(start)
            goal_poly = vehicle.get_visualization_polygon(goal)
            
            ax.add_patch(Polygon(start_poly, closed=True, facecolor='green', alpha=0.5))
            ax.add_patch(Polygon(goal_poly, closed=True, facecolor='red', alpha=0.5))
            
            # 4. 设置标题和标签
            if row_idx == 0:
                ax.set_title(f"Density: {density}", fontsize=14, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"{v_name}\nY [m]", fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel("Y [m]")
            
            ax.set_xlabel("X [m]")
            ax.grid(True, linestyle=':', alpha=0.3)
            
            # 在图内标注简要信息
            ax.text(0.02, 0.95, f"Res: {resolution}m", transform=ax.transAxes, 
                    color='blue', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

    # --- 5. 保存与显示 ---
    plt.tight_layout()
    
    output_filename = "map_generation_comparison.png"
    print(f"Saving comparison result to {output_filename}...")
    plt.savefig(output_filename, dpi=150) # 先保存！
    
    plt.show() # 后显示

if __name__ == "__main__":
    test_map_generation_comparison()