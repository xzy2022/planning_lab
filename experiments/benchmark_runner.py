# experiments\benchmark_runner.py
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

# --- 配置日志目录 ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

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

def save_failure_snapshot(grid_map, start, goal, debugger, algo_name, density, trial_idx):
    """
    当规划失败时，保存当前的地图和探索状态截图
    """
    # 创建图形，不显示在屏幕上 (off-screen)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. 绘制地图背景
    ax.imshow(grid_map.data, cmap='Greys', origin='lower', 
              extent=[0, grid_map.width * grid_map.resolution, 
                      0, grid_map.height * grid_map.resolution],
              alpha=0.5)
    
    # 2. 绘制起终点
    ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
    ax.plot(goal.x, goal.y, 'rx', markersize=10, label='Goal')
    
    # 3. 绘制已探索节点
    if debugger.expanded_nodes:
        nodes = np.array(debugger.expanded_nodes)
        if nodes.ndim == 2 and nodes.shape[0] > 0:
            ax.scatter(nodes[:, 0], nodes[:, 1], c='red', s=2, alpha=0.6, label='Expanded Nodes')
    
    ax.set_title(f"FAILURE LOG: {algo_name} | Density: {density} | Trial: {trial_idx}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_aspect('equal')
    
    filename = f"fail_{algo_name}_d{density}_t{trial_idx}.png"
    filepath = os.path.join(LOG_DIR, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"  [LOG] {algo_name} 规划失败截图已保存: {filepath}")

def run_experiment():
    # --- 1. 实验参数设置 ---W
    densities = [0.05, 0.1, 0.15, 0.2, 0.25]
    num_trials = 10
    map_width, map_height, res = 100, 100, 0.5
    
    start_state = State(2.0, 2.0, 0.0)
    goal_state = State(48.0, 48.0, 0.0)

    # 车辆配置
    vehicle_config = PointMassConfig(width=1.0, length=1.0, safe_margin=0.1)
    vehicle = PointMassVehicle(vehicle_config)

    # 生成地图用的大车配置 (确保有足够空间)
    vehicle_big_config = PointMassConfig(width=2.0, length=2.0, safe_margin=0.1)
    vehicle_big = PointMassVehicle(vehicle_big_config)
    
    results = []

    print(f"{'Density':<8} | {'Algo':<5} | {'Succ%':<6} | {'Time(ms)':<10} | {'T_Std':<8} | {'Nodes':<8} | {'N_Std':<8}")
    print("-" * 85)

    # --- 2. 循环实验 ---
    for density in densities:
        
        stats = {
            'A*': {'success': 0, 'time': [], 'nodes': [], 'length': []},
            'RRT': {'success': 0, 'time': [], 'nodes': [], 'length': []}
        }

        for i in range(num_trials):
            seed = 42 + i + int(density * 1000)
            grid_map = GridMap(width=map_width, height=map_height, resolution=res)
            
            generator = MapGenerator(obstacle_density=density, inflation_radius_m=0.5, seed=seed)
            generator.generate(grid_map, vehicle, start_state, goal_state, extra_paths=5, dead_ends=2)
            
            col_config = CollisionConfig(method=CollisionMethod.RASTER)
            checker = CollisionChecker(col_config, vehicle, grid_map)

            # --- B. 运行 A* ---
            heuristic = OctileHeuristic()
            dist_cost = DistanceCost()
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
            else:
                save_failure_snapshot(grid_map, start_state, goal_state, debugger_astar, "AStar", density, i)

            # --- C. 运行 RRT ---
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
            else:
                save_failure_snapshot(grid_map, start_state, goal_state, debugger_rrt, "RRT", density, i)

        # --- 3. 汇总与计算标准差 ---
        for algo in ['A*', 'RRT']:
            succ_rate = (stats[algo]['success'] / num_trials) * 100
            
            # 计算均值和标准差
            times = stats[algo]['time']
            nodes = stats[algo]['nodes']
            lengths = stats[algo]['length']

            avg_time = np.mean(times) if times else 0
            std_time = np.std(times) if times else 0
            
            avg_nodes = np.mean(nodes) if nodes else 0
            std_nodes = np.std(nodes) if nodes else 0
            
            avg_len = np.mean(lengths) if lengths else 0
            std_len = np.std(lengths) if lengths else 0
            
            # 打印简报
            print(f"{density:<8.2f} | {algo:<5} | {succ_rate:<6.1f} | {avg_time:<10.2f} | {std_time:<8.2f} | {avg_nodes:<8.1f} | {std_nodes:<8.1f}")
            
            results.append({
                'Density': density,
                'Algorithm': algo,
                'SuccessRate': succ_rate,
                
                'TimeMean': avg_time,
                'TimeStd': std_time,
                
                'NodesMean': avg_nodes,
                'NodesStd': std_nodes,
                
                'LengthMean': avg_len,
                'LengthStd': std_len
            })

    return pd.DataFrame(results)

def plot_comparisons(df):
    """可视化对比图表 (包含标准差误差棒) - 2x2 布局"""
    # 使用 2行 2列，调整 figsize 为更接近正方形/竖向的比例
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 展平 axes 数组以便于在循环中线性索引 (从 [[ax1, ax2], [ax3, ax4]] 变为 [ax1, ax2, ax3, ax4])
    axes_flat = axes.flatten()
    
    # 格式: (Y轴字段名, 标准差字段名, Y轴标签, 标题)
    metrics = [
        ('SuccessRate', None, 'Success Rate (%)', 'Reliability'),
        ('TimeMean', 'TimeStd', 'Time (ms)', 'Time Complexity (Mean ± Std)'),
        ('NodesMean', 'NodesStd', 'Expanded Nodes', 'Space Complexity (Mean ± Std)'),
        ('LengthMean', 'LengthStd', 'Path Length (m)', 'Optimality (Mean ± Std)')
    ]
    
    # 定义颜色和样式
    styles = {
        'A*': {'color': 'blue', 'fmt': 'o-', 'ecolor': 'lightblue'},
        'RRT': {'color': 'orange', 'fmt': 's-', 'ecolor': 'moccasin'}
    }
    
    for i, (metric, std_metric, ylabel, title) in enumerate(metrics):
        ax = axes_flat[i] # 使用展平后的索引
        
        for algo in ['A*', 'RRT']:
            data = df[df['Algorithm'] == algo]
            x = data['Density']
            y = data[metric]
            
            style = styles[algo]
            
            if std_metric:
                yerr = data[std_metric]
                ax.errorbar(x, y, yerr=yerr, 
                            label=algo,
                            fmt=style['fmt'],       
                            color=style['color'],   
                            ecolor=style['ecolor'], 
                            elinewidth=2,           
                            capsize=5,              
                            capthick=2,
                            alpha=0.9)
            else:
                ax.plot(x, y, style['fmt'], label=algo, color=style['color'])
        
        ax.set_xlabel('Obstacle Density', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 仅在第一个图显示图例
        if i == 0:
            ax.legend(fontsize=10)
        
    plt.tight_layout()
    output_file = "benchmark_with_std.png"
    print(f"\nSaving plot to {output_file}...")
    plt.savefig(output_file, dpi=150)
    plt.show()

if __name__ == "__main__":
    print(f"=== 开始 PointMass 路径规划对比实验 (A* vs RRT) ===")
    print(f"=== 失败日志将保存至: {LOG_DIR} ===")
    
    df_results = run_experiment()
    print("\n实验结束，正在绘图...")
    plot_comparisons(df_results)