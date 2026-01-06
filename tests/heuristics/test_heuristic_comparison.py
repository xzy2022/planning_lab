# tests/heuristics/test_heuristic_comparison.py
import sys
import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. 路径环境设置 ---
# 假设当前脚本在 tests/heuristics/ 下，向上回溯 3 层找到项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.point_mass import PointMassVehicle, PointMassConfig
from src.types import State
from src.collision.checker import CollisionChecker, CollisionConfig, CollisionMethod
from src.planning.planners import AStarPlanner
from src.planning.costs import DistanceCost
from src.visualization.debugger import PlanningDebugger

# 导入已有启发式
from src.planning.heuristics import EuclideanHeuristic, OctileHeuristic, ZeroHeuristic, ManhattanHeuristic



# --- 3. 辅助函数 ---

LOG_DIR = "logs/"
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

def save_failure_snapshot(grid_map, start, goal, debugger, h_name, density, trial_idx):
    """保存失败现场截图"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid_map.data, cmap='Greys', origin='lower', 
              extent=[0, grid_map.width * grid_map.resolution, 
                      0, grid_map.height * grid_map.resolution],
              alpha=0.5)
    ax.plot(start.x, start.y, 'go', label='Start')
    ax.plot(goal.x, goal.y, 'rx', label='Goal')
    
    if debugger.expanded_nodes:
        nodes = np.array(debugger.expanded_nodes)
        if nodes.ndim == 2 and nodes.shape[0] > 0:
            ax.scatter(nodes[:, 0], nodes[:, 1], c='red', s=1, alpha=0.3, label='Expanded')
            
    ax.set_title(f"FAIL: {h_name} | D={density} | T={trial_idx}")
    filename = f"fail_{h_name}_d{density}_t{trial_idx}.png"
    plt.savefig(os.path.join(LOG_DIR, filename))
    plt.close(fig)

# --- 4. 实验核心逻辑 ---

def run_heuristic_experiment():
    # 实验配置
    densities = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3] # 障碍物密度梯度
    num_trials = 50                  # 每个密度下的测试次数
    map_width, map_height, res = 100, 100, 0.5 # 50m x 50m
    
    start_state = State(2.0, 2.0, 0.0)
    goal_state = State(48.0, 48.0, 0.0)

    # 车辆配置
    vehicle_config = PointMassConfig(width=1.0, length=1.0, safe_margin=0.1)
    vehicle = PointMassVehicle(vehicle_config)
    # 用于生成地图的大车（确保有宽敞路径）
    vehicle_big = PointMassVehicle(PointMassConfig(width=2.0, length=2.0, safe_margin=0.1))

    # 待测启发式列表
    heuristics_to_test = {
        'Dijkstra (h=0)': ZeroHeuristic(),
        'Euclidean': EuclideanHeuristic(),
        'Octile': OctileHeuristic(),
        'Manhattan (Inadmissible)': ManhattanHeuristic()
    }

    results = []
    
    print(f"{'Density':<8} | {'Heuristic':<25} | {'Succ%':<6} | {'Time(ms)':<8} | {'Nodes':<8} | {'Len(m)':<8}")
    print("-" * 100)

    for density in densities:
        # 每个密度下，初始化统计容器
        stats = {name: {'success': 0, 'time': [], 'nodes': [], 'length': []} 
                 for name in heuristics_to_test.keys()}

        for i in range(num_trials):
            # A. 生成环境 (固定种子，确保所有启发式在同一张地图上跑)
            seed = 100 + int(density * 100) + i
            grid_map = GridMap(width=map_width, height=map_height, resolution=res)
            
            generator = MapGenerator(obstacle_density=density, inflation_radius_m=0.5, seed=seed)
            # 确保至少有一条可行路径
            generator.generate(grid_map, vehicle_big, start_state, goal_state, extra_paths=1, dead_ends=2)
            
            # 初始化碰撞检测 (Raster 模式最快)
            col_config = CollisionConfig(method=CollisionMethod.RASTER)
            checker = CollisionChecker(col_config, vehicle, grid_map)
            
            # B. 遍历所有启发式进行规划
            for h_name, h_func in heuristics_to_test.items():
                
                # 构造规划器
                planner = AStarPlanner(
                    vehicle_model=vehicle,
                    collision_checker=checker,
                    heuristic=h_func,
                    cost_functions=[DistanceCost()],
                    weights=[1.0]
                )
                debugger = PlanningDebugger()

                # 执行规划 & 计时
                t0 = time.perf_counter()
                path = planner.plan(start_state, goal_state, grid_map, debugger)
                t1 = time.perf_counter()

                # 记录数据
                if path:
                    stats[h_name]['success'] += 1
                    stats[h_name]['time'].append((t1 - t0) * 1000)
                    stats[h_name]['nodes'].append(len(debugger.expanded_nodes))
                    stats[h_name]['length'].append(calculate_path_length(path))
                else:
                    # 记录失败
                    if i < 2: # 仅保存少量截图防止刷屏
                        save_failure_snapshot(grid_map, start_state, goal_state, debugger, h_name, density, i)

        # 汇总当前 Density 的数据
        for h_name in heuristics_to_test.keys():
            s_data = stats[h_name]
            succ_rate = (s_data['success'] / num_trials) * 100
            
            avg_time = np.mean(s_data['time']) if s_data['time'] else 0
            std_time = np.std(s_data['time']) if s_data['time'] else 0
            
            avg_nodes = np.mean(s_data['nodes']) if s_data['nodes'] else 0
            std_nodes = np.std(s_data['nodes']) if s_data['nodes'] else 0
            
            avg_len = np.mean(s_data['length']) if s_data['length'] else 0
            std_len = np.std(s_data['length']) if s_data['length'] else 0
            
            print(f"{density:<8.1f} | {h_name:<25} | {succ_rate:<6.0f} | {avg_time:<8.1f} | {avg_nodes:<8.0f} | {avg_len:<8.2f}")

            results.append({
                'Density': density,
                'Heuristic': h_name,
                'SuccessRate': succ_rate,
                'TimeMean': avg_time, 'TimeStd': std_time,
                'NodesMean': avg_nodes, 'NodesStd': std_nodes,
                'LengthMean': avg_len, 'LengthStd': std_len
            })

    return pd.DataFrame(results)

# --- 5. 可视化绘图 ---

def plot_heuristic_comparison(df):
    """绘制 2x2 对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metrics = [
        ('SuccessRate', None, 'Success Rate (%)', 'Robustness'),
        ('TimeMean', 'TimeStd', 'Computation Time (ms)', 'Speed (Lower is Better)'),
        ('NodesMean', 'NodesStd', 'Expanded Nodes', 'Search Efficiency (Lower is Better)'),
        ('LengthMean', 'LengthStd', 'Path Length (m)', 'Optimality (Lower is Better)')
    ]
    
    # 定义绘图样式
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']
    
    heuristics = df['Heuristic'].unique()
    
    for i, (metric, std_metric, ylabel, title) in enumerate(metrics):
        ax = axes[i]
        
        for idx, h_name in enumerate(heuristics):
            data = df[df['Heuristic'] == h_name]
            
            x = data['Density']
            y = data[metric]
            
            kwargs = {
                'label': h_name,
                'marker': markers[idx % len(markers)],
                'linestyle': linestyles[idx % len(linestyles)],
                'linewidth': 2,
                'alpha': 0.8
            }
            
            if std_metric:
                yerr = data[std_metric]
                ax.errorbar(x, y, yerr=yerr, capsize=4, **kwargs)
            else:
                ax.plot(x, y, **kwargs)
                
        ax.set_xlabel('Obstacle Density')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 仅在第一个图显示图例
        if i == 0:
            ax.legend()

    plt.suptitle("Impact of Heuristics on A* Performance", fontsize=16)
    plt.tight_layout()
    
    output_path = os.path.join(LOG_DIR, "heuristic_comparison.png")
    print(f"\nSaving plot to {output_path}...")
    plt.savefig(output_path, dpi=150)
    plt.show()

if __name__ == "__main__":
    print("=== A* 启发式性能对比测试 ===")
    print("对比项: Dijkstra vs Euclidean vs Octile vs Manhattan")
    
    df_results = run_heuristic_experiment()
    
    print("\n实验完成，正在绘图...")
    plot_heuristic_comparison(df_results)