# tests/map/test_map_generation.py
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon  # [新增] 用于画车身轮廓

# 路径黑魔法
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.ackermann import AckermannVehicle
from src.vehicles.config import AckermannConfig
from src.vehicles.point_mass import PointMassVehicle, PointMassConfig
from src.types import State

def test_feasible_map_generation():
    # 1. 准备组件
    # [优化] 提高分辨率以观察 Footprint 效果
    # 100m x 100m 地图:
    # 旧: 200 x 0.5m = 100m (粗糙)
    # 新: 1000 x 0.1m = 100m (精细，能看清车身锯齿)
    grid_map = GridMap(100, 100, resolution=0.5)
    
    # config = AckermannConfig(
    #     wheelbase=2.5, 
    #     max_steer_deg=34.4,
    #     width=2.0  # 显式确认车宽，便于观察
    # )
    # vehicle = AckermannVehicle(config)

    # 车辆配置 (质点模型)
    vehicle_config = PointMassConfig(width=1.0, length=1.0, safe_margin=0.1)
    vehicle = PointMassVehicle(vehicle_config)

    # 车辆配置 (质点模型) 用于获得非狭窄空间
    vehicle_big_config = PointMassConfig(width=2.0, length=2.0, safe_margin=0.1)
    vehicle_big = PointMassVehicle(vehicle_big_config)
    
    start = State(2.0, 2.0, 0.0)
    goal = State(48.0, 48.0, 0.0)
    
    # 2. 调用生成器
    # 计算物理尺寸用于打印和可视化
    phys_width = grid_map.width * grid_map.resolution
    phys_height = grid_map.height * grid_map.resolution

    print(f"Map Physical Size: {phys_width:.1f}m x {phys_height:.1f}m")
    print(f"Map Resolution: {grid_map.resolution}m")
    print("生成包含可行路径的地图中 (使用 Footprint 推土机)...")
    
    # 1. 实例化生成器
    # 提示：由于现在是贴合车身清除，obstacle_density 可以适当调高一点点测试极限，
    # 或者保持不变观察生成的“窄通道”
    obstacle_density = 0.1
    generator = MapGenerator(
        obstacle_density=obstacle_density, 
        inflation_radius_m=0.5, # 这是障碍物的膨胀，不是车身的
        num_waypoints=5
    )
    
    # 2. 执行生成 (Generator 内部会自动处理“胖”车身逻辑)
    # extra_paths=1: 至少会有两条路通往终点
    # dead_ends=5: 生成5条乱七八糟的干扰路径
    generator.generate(grid_map, vehicle, start, goal, extra_paths=1, dead_ends=5)
    
    # 3. 可视化
    fig, ax = plt.subplots(figsize=(12, 12)) #稍微调大画布
    
    # [关键] 使用 extent 参数设置物理坐标范围
    map_extent = [0, phys_width, 0, phys_height]

    # A. 画地图 
    ax.imshow(grid_map.data, cmap='Greys', origin='lower', extent=map_extent)
    
    # B. 画起点和终点 (不仅仅是点，我们把车画出来，看是否贴合)
    ax.plot(start.x, start.y, 'go', markersize=5, label='Start Center')
    ax.plot(goal.x, goal.y, 'rx', markersize=5, label='Goal Center')

    # [新增] 画出起点和终点的车身轮廓，验证 Generator 是否清除了足够的空间
    start_poly = vehicle.get_visualization_polygon(start)
    goal_poly = vehicle.get_visualization_polygon(goal)
    
    ax.add_patch(Polygon(start_poly, closed=True, facecolor='green', alpha=0.5, label='Start Body'))
    ax.add_patch(Polygon(goal_poly, closed=True, facecolor='red', alpha=0.5, label='Goal Body'))
    
    ax.set_title(f"Feasible Map Generation (Footprint Mode)\n"
                 f"Density: {obstacle_density}, Resolution: {grid_map.resolution}m\n"
                 f"Notice: Path should fit the rectangular body shape")
    
    ax.set_xlabel("X Position [m]")
    ax.set_ylabel("Y Position [m]")
    
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    # 聚焦显示（可选）：如果你想看细节，可以取消下面注释只看局部
    # ax.set_xlim(0, 30)
    # ax.set_ylim(0, 30)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_feasible_map_generation()