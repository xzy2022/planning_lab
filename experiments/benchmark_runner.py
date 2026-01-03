import sys
import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 路径设置 ---
# 确保能找到 src 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.point_mass import PointMassVehicle, PointMassConfig
from src.types import State
from src.collision.checker import CollisionChecker, CollisionConfig, CollisionMethod
from src.planning.planners import AStarPlanner, RRTPlanner
from src.planning.heuristics import OctileHeuristic
from src.planning.costs import DistanceCost
from src.visualization.debugger import PlanningDebugger

def calculate_path_length(path):
    """计算路径的累积欧氏距离"""
    if not path or len(path) < 2:
        return 0.0
    length = 0.0
    for i in range(len(path) - 1):
        dx = path[i+1].x - path[i].x
        dy = path[i+1].y - path[i].y
        length += math.hypot(dx, dy)
    return length

def run_experiment():
    # --- 1. 实验参数设置 ---
    densities = [0.05, 0.1, 0.15, 0.2, 0.25] # 障碍物密度梯度
    num_trials = 10                          # 每个密度下测试多少张地图
    map_width, map_height, res = 100, 100, 0.5 # 50m x 50m 地图
    
    # 起终点设置
    start_state = State(2.0, 2.0, 0.0)
    goal_state = State(48.0, 48.0, 0.0)

    # 车辆配置 (质点模型)
    vehicle_config = PointMassConfig(width=1.0, length=1.0, safe_margin=0.1)
    vehicle = PointMassVehicle(vehicle_config)
    
    # 结果存储容器
    results = []

    print(f"{'Density':<10} | {'Algo':<10} | {'Success%':<10} | {'Time(ms)':<10} | {'Nodes':<10} | {'Len(m)':<10}")
    print("-" * 80)

    # --- 2. 循环实验 ---
    for density in densities:
        
        # 统计变量
        stats = {
            'A*': {'success': 0, 'time': [], 'nodes': [], 'length': []},
            'RRT': {'success': 0, 'time': [], 'nodes': [], 'length': []}
        }

        for i in range(num_trials):
            # A. 生成地图 (使用相同的 Seed 确保 A* 和 RRT 在同一张图上跑)
            seed = 42 + i + int(density * 1000)
            grid_map = GridMap(width=map_width, height=map_height, resolution=res)
            
            # 使用 MapGenerator 保证至少有一条路 (通过 Carve 机制)
            generator = MapGenerator(obstacle_density=density, inflation_radius_m=0.5, seed=seed)
            # 生成时使用 PointMass 确保路径匹配车辆尺寸
            generator.generate(grid_map, vehicle, start_state, goal_state, extra_paths=0, dead_ends=2)
            
            # 初始化碰撞检测器
            col_config = CollisionConfig(method=CollisionMethod.RASTER)
            checker = CollisionChecker(col_config, vehicle, grid_map)

            # --- B. 运行 A* ---
            heuristic = OctileHeuristic()
            dist_cost = DistanceCost()
            # 权重设置为 1.0 (保证最优性)
            a_star = AStarPlanner(vehicle, checker, heuristic, [dist_cost], [1.0])
            debugger_astar = PlanningDebugger()
            
            t0 = time.perf_counter()
            path_astar = a_star.plan(start_state, goal_state, grid_map, debugger_astar)
            t1 = time.perf_counter()
            
            if path_astar:
                stats['A*']['success'] += 1
                stats['A*']['time'].append((t1 - t0) * 1000)
                stats['A*']['nodes'].append(len(debugger_astar.expanded_nodes))
                stats['A*']['length'].append(calculate_path_length(path_astar))

            # --- C. 运行 RRT ---
            # Step size 设为 2.0m, Max Iter 设为 5000
            rrt = RRTPlanner(vehicle, checker, step_size=2.0, max_iterations=5000, goal_sample_rate=0.1)
            debugger_rrt = PlanningDebugger()
            
            t0 = time.perf_counter()
            path_rrt = rrt.plan(start_state, goal_state, grid_map, debugger_rrt)
            t1 = time.perf_counter()

            if path_rrt:
                stats['RRT']['success'] += 1
                stats['RRT']['time'].append((t1 - t0) * 1000)
                stats['RRT']['nodes'].append(len(debugger_rrt.expanded_nodes))
                stats['RRT']['length'].append(calculate_path_length(path_rrt))

        # --- 3. 汇总当前 Density 的数据 ---
        for algo in ['A*', 'RRT']:
            succ_rate = (stats[algo]['success'] / num_trials) * 100
            avg_time = np.mean(stats[algo]['time']) if stats[algo]['time'] else 0
            avg_nodes = np.mean(stats[algo]['nodes']) if stats[algo]['nodes'] else 0
            avg_len = np.mean(stats[algo]['length']) if stats[algo]['length'] else 0
            
            # 打印
            print(f"{density:<10.2f} | {algo:<10} | {succ_rate:<10.1f} | {avg_time:<10.2f} | {avg_nodes:<10.1f} | {avg_len:<10.2f}")
            
            results.append({
                'Density': density,
                'Algorithm': algo,
                'SuccessRate': succ_rate,
                'TimeMean': avg_time,
                'NodesMean': avg_nodes,
                'LengthMean': avg_len
            })

    return pd.DataFrame(results)

def plot_comparisons(df):
    """可视化对比图表"""
    # 准备画布: 修改为 1行4列，且增加宽度 (24, 5)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    # 指标列表：新增 SuccessRate
    metrics = [
        ('SuccessRate', 'Success Rate (%)', 'Reliability'),
        ('TimeMean', 'Computation Time (ms)', 'Time Complexity'),
        ('NodesMean', 'Expanded Nodes', 'Space Complexity'),
        ('LengthMean', 'Path Length (m)', 'Optimality')
    ]
    
    densities = df['Density'].unique()
    
    for i, (metric, ylabel, title) in enumerate(metrics):
        ax = axes[i]
        
        # 提取数据
        data_astar = df[df['Algorithm'] == 'A*']
        data_rrt = df[df['Algorithm'] == 'RRT']
        
        # 绘图
        ax.plot(data_astar['Density'], data_astar[metric], 'o-', label='A*', color='blue')
        ax.plot(data_rrt['Density'], data_rrt[metric], 's-', label='RRT', color='orange')
        
        ax.set_xlabel('Obstacle Density')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 仅在第一个图显示图例，避免遮挡
        if i == 0:
            ax.legend()
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== 开始 PointMass 路径规划对比实验 (A* vs RRT) ===")
    df_results = run_experiment()
    print("\n实验结束，正在绘图...")
    plot_comparisons(df_results)