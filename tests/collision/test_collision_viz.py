import sys
import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

# --- 1. 路径设置 (确保能导入 src) ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.vehicles.ackermann import AckermannVehicle, AckermannConfig
from src.vehicles.base import State
from src.map.grid_map import GridMap
# [新增] 导入你已经实现的模块
from src.collision.checker import CollisionChecker
from src.collision.config import CollisionConfig, CollisionMethod

def setup_environment():
    """准备地图和车辆"""
    # 1. 创建一个小地图 (10m x 10m)
    grid_map = GridMap(width=100, height=100, resolution=0.1)
    
    # 2. 在地图中间放置一个矩形障碍物 (模拟墙壁或柱子)
    for y in range(40, 61):
        for x in range(50, 56):
            grid_map.data[y, x] = 1
            
    # 3. 初始化车辆
    vehicle_config = AckermannConfig()
    vehicle = AckermannVehicle(vehicle_config)
    
    return grid_map, vehicle

def draw_map(ax, grid_map):
    """画底图障碍物"""
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
        
        # 2. 临界状态: 外接圆碰到了，但车身没碰到
        State(x=3.2, y=4.95, theta_rad=math.radians(90)), 
        
        # 3. 碰撞状态: 车头插入障碍物
        State(x=3.9, y=5.0, theta_rad=math.radians(90)),
        
        # 4. 复杂状态: 斜着停 (测试光栅化的锯齿效应)
        State(x=3.5, y=3.5, theta_rad=math.radians(45))
    ]

    # --- 定义要测试的检测模式 (包含新增的 RASTER) ---
    methods = [
        (CollisionMethod.CIRCLE_ONLY, "Circle Only"),
        (CollisionMethod.MULTI_CIRCLE, "Multi-Circle"),
        (CollisionMethod.POLYGON, "Polygon SAT"),
        (CollisionMethod.RASTER, "Raster/Footprint") # [新增]
    ]

    # --- 性能统计容器 ---
    # 格式: { method_name: [time_us_1, time_us_2, ...] }
    perf_stats = {name: [] for _, name in methods}

    # --- 开始绘图 ---
    fig, axes = plt.subplots(len(test_states), len(methods), figsize=(16, 12))
    
    # 维度处理
    if len(test_states) == 1: axes = np.array([axes])
    if len(methods) == 1: axes = axes[:, np.newaxis]

    print(f"{'Method':<20} | {'Status':<10} | {'Time (us)':<10}")
    print("-" * 50)

    for row, state in enumerate(test_states):
        for col, (method, method_name) in enumerate(methods):
            ax = axes[row, col]
            
            # 1. 绘制地图背景
            draw_map(ax, grid_map)
            
            # 2. 初始化检测器
            config = CollisionConfig(method=method)
            checker = CollisionChecker(config, vehicle=vehicle, grid_map=grid_map)
            
            # 3. 执行检测并计时
            # [可选] 预热一次，防止首次加载缓存或 JIT 影响
            _ = checker.check(vehicle, state, grid_map)
            
            start_ns = time.perf_counter_ns()
            is_collision = checker.check(vehicle, state, grid_map)
            end_ns = time.perf_counter_ns()
            
            duration_us = (end_ns - start_ns) / 1000.0
            perf_stats[method_name].append(duration_us)
            
            print(f"{method_name:<20} | {'COLLISION' if is_collision else 'SAFE':<10} | {duration_us:.2f}")
            
            # 4. 结果标题
            color = 'red' if is_collision else 'green'
            result_text = "COLLISION" if is_collision else "SAFE"
            ax.set_title(f"{method_name}\n{result_text} ({duration_us:.1f} us)", 
                         color=color, fontweight='bold', fontsize=9)
            
            # 5. 可视化几何体
            # 总是画出车身轮廓作为参考 (蓝色虚线)
            vis_poly = vehicle.get_visualization_polygon(state)
            ax.add_patch(Polygon(vis_poly, closed=True, fill=False, edgecolor='blue', linestyle='--', alpha=0.5))
            
            # --- 分类可视化 ---
            if method == CollisionMethod.CIRCLE_ONLY:
                bx, by, br = vehicle.get_bounding_circle(state)
                fill_alpha = 0.2 if is_collision else 0.0
                ax.add_patch(Circle((bx, by), br, color=color, alpha=fill_alpha, fill=True))
                ax.add_patch(Circle((bx, by), br, edgecolor=color, fill=False))

            elif method == CollisionMethod.MULTI_CIRCLE:
                cxs, cys, r = vehicle.get_collision_circles(state)
                for cx, cy in zip(cxs, cys):
                    ax.add_patch(Circle((cx, cy), r, color=color, alpha=0.3))
                # 辅助: Broad Phase
                bx, by, br = vehicle.get_bounding_circle(state)
                ax.add_patch(Circle((bx, by), br, edgecolor='gray', linestyle=':', fill=False))

            elif method == CollisionMethod.POLYGON:
                poly_points = vehicle.get_collision_polygon(state)
                ax.add_patch(Polygon(poly_points, closed=True, facecolor=color, alpha=0.3))
                # 辅助: Broad Phase
                bx, by, br = vehicle.get_bounding_circle(state)
                ax.add_patch(Circle((bx, by), br, edgecolor='gray', linestyle=':', fill=False))

            elif method == CollisionMethod.RASTER:
                # [关键更新] 可视化 Raster 占据的网格
                # 假设 checker.footprint_model 是公开属性
                if hasattr(checker, 'footprint_model') and checker.footprint_model:
                    indices = checker.footprint_model.get_occupied_indices(state)
                    res = grid_map.resolution
                    
                    # 绘制所有占据的格子
                    for (ix, iy) in indices:
                        # 仅绘制在视野内的，优化绘图速度
                        if 0 <= ix < grid_map.width and 0 <= iy < grid_map.height:
                            # 转换为地图坐标 (左下角)
                            rect_x = ix * res
                            rect_y = iy * res
                            ax.add_patch(Rectangle((rect_x, rect_y), res, res, facecolor=color, alpha=0.5))
                else:
                    # 如果无法获取模型用于绘图，退化为画多边形
                    ax.text(state.x, state.y, "No Footprint Model Access", fontsize=8)

            # 设置视野
            ax.set_aspect('equal')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.grid(True, linestyle=':', alpha=0.3)
            
            if col == 0:
                ax.set_ylabel(f"State {row+1}\n({state.x:.1f}, {state.y:.1f})")

    plt.tight_layout()
    plt.show()

    # --- 打印统计结果 ---
    print("\n" + "="*65)
    print(f"{'Method':<20} | {'Mean Time (us)':<15} | {'Std Dev':<10} | {'Performance'}")
    print("-" * 65)
    
    # 获取 Raster 的均值作为基准
    raster_times = perf_stats.get("Raster/Footprint", [])
    raster_mean = np.mean(raster_times) if raster_times else 0.0
    
    for name, times in perf_stats.items():
        if not times: continue
        mean_t = np.mean(times)
        std_t = np.std(times)
        
        # 计算相对于 Raster 的慢倍数
        ratio_str = ""
        if raster_mean > 0:
            ratio = mean_t / raster_mean
            if ratio < 0.99: ratio_str = f"({1/ratio:.1f}x faster)"
            elif ratio > 1.01: ratio_str = f"({ratio:.1f}x slower)"
            else: ratio_str = "(Baseline)"
            
        print(f"{name:<20} | {mean_t:<15.2f} | {std_t:<10.2f} | {ratio_str}")
    print("="*65)

if __name__ == "__main__":
    visualize_collision_test()