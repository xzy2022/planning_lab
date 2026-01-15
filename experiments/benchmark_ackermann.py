import sys
import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Path Setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.ackermann import AckermannVehicle
from src.vehicles.config import AckermannConfig
from src.types import State
from src.collision.checker import CollisionChecker, CollisionConfig, CollisionMethod
from src.planning.planners import RRTPlanner, HybridAStarPlanner
from src.planning.smoother import GreedyShortcutSmoother
from src.visualization.debugger import PlanningDebugger


# --- Log Directory ---
# Configured in BenchmarkConfig
from experiments.benchmark_config import BenchmarkConfig as cfg
os.makedirs(cfg.LOG_DIR, exist_ok=True)

def log_experiment(msg, to_console=True):
    with open(cfg.EXPERIMENT_LOG_PATH, "a") as f:
        f.write(msg + "\n")
    if to_console:
        print(msg)

def calculate_path_length(path):
    """Calculate cumulative Euclidean distance of the path (only x,y)"""
    if not path or len(path) < 2:
        return 0.0
    length = 0.0
    for i in range(len(path) - 1):
        dx = path[i+1].x - path[i].x
        dy = path[i+1].y - path[i].y
        length += math.hypot(dx, dy)
    return length

def save_failure_snapshot(grid_map, start, goal, debugger, algo_name, density, trial_idx):
    """Save snapshot when planning fails"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Background
    ax.imshow(grid_map.data, cmap='Greys', origin='lower', 
              extent=[0, grid_map.width * grid_map.resolution, 
                      0, grid_map.height * grid_map.resolution],
              alpha=0.5)
    
    # Start/Goal
    ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
    ax.plot(goal.x, goal.y, 'rx', markersize=10, label='Goal')
    
    # Expanded Nodes (if available)
    if debugger.expanded_nodes:
        nodes = np.array(debugger.expanded_nodes)
        if nodes.ndim == 2 and nodes.shape[0] > 0:
            ax.scatter(nodes[:, 0], nodes[:, 1], c='red', s=2, alpha=0.6, label='Expanded Nodes')
    
    ax.set_title(f"FAILURE: {algo_name} | Density: {density} | Trial: {trial_idx}")
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    
    filename = f"fail_ackermann_{algo_name}_d{density}_t{trial_idx}.png"
    filepath = os.path.join(cfg.LOG_DIR, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"  [LOG] Failure snapshot saved: {filepath}")
    
    # 打印复现命令 并 记录详细信息到文件
    repro_cmd = f"python experiments/debug_experiment.py --algo {algo_name} --density {density} --seed {grid_map.seed}"
    print(f"  [REPRODUCE] {repro_cmd}")
    
    # 获取详细配置
    from experiments.benchmark_config import BenchmarkConfig as cfg
    algo_config = {}
    if algo_name == "RRT":
        algo_config = cfg.RRT_PARAMS
    elif algo_name == "HybridAStar":
        algo_config = cfg.HYBRID_ASTAR_PARAMS
        
    # 统计信息
    num_nodes = len(debugger.expanded_nodes) if hasattr(debugger, 'expanded_nodes') else 0
    
    log_msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FAILURE: {algo_name} | Density: {density} | Trial: {trial_idx}\n"
    log_msg += f"  Reproduction: {repro_cmd}\n"
    log_msg += f"  Context:\n"
    log_msg += f"    Map Seed: {grid_map.seed}\n"
    log_msg += f"    Start: {start} | Goal: {goal}\n"
    log_msg += f"    Expanded Nodes: {num_nodes}\n"
    log_msg += f"    Config: {algo_config}\n"
    log_msg += "-" * 80
    
    log_experiment(log_msg, to_console=False)

from experiments.benchmark_config import BenchmarkConfig

def run_experiment():
    # --- 1. Experiment Parameters ---
    cfg = BenchmarkConfig
    
    # Vehicle & Plow Configs handled in config class
    vehicle = AckermannVehicle(cfg.VEHICLE_CONFIG)
    plow_vehicle = AckermannVehicle(cfg.PLOW_CONFIG)
    
    # Map Generator Vehicle (Bulldozer)
    # Slightly larger to ensure paths exist - Configured in BenchmarkConfig

    # Collision Checker (Polygon for accuracy)
    # Configured in BenchmarkConfig
    
    # --- Initialize Log ---
    if os.path.exists(cfg.EXPERIMENT_LOG_PATH):
        os.remove(cfg.EXPERIMENT_LOG_PATH)
        
    log_experiment("=== Ackermann Planner Benchmark Experiment Log ===")
    log_experiment(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_experiment("Configuration:")
    log_experiment(f"  Map: {cfg.PHYS_WIDTH}x{cfg.PHYS_HEIGHT}m (Res: {cfg.RESOLUTION}m)")
    log_experiment(f"  Densities: {cfg.DENSITIES}")
    log_experiment(f"  Trials/Density: {cfg.NUM_TRIALS}")
    log_experiment("-" * 80)
    
    results = []

    header = f"{'Density':<8} | {'Algo':<12} | {'Succ%':<6} | {'Time(ms)':<15} | {'Len(m)':<15} | {'Nodes':<15}"
    log_experiment(header)
    log_experiment("-" * 85)

    # --- 2. Loop Execution ---
    for density in cfg.DENSITIES:
        
        stats = {
            'RRT': {'success': 0, 'time': [], 'nodes': [], 'length': []},
            'RRT+Smooth': {'success': 0, 'time': [], 'nodes': [], 'length': []},
            'HybridA*': {'success': 0, 'time': [], 'nodes': [], 'length': []}
        }

        for i in range(cfg.NUM_TRIALS):
            seed = cfg.RANDOM_SEED_BASE + int(density * 100) + i
            grid_map = GridMap(width=cfg.MAP_WIDTH, height=cfg.MAP_HEIGHT, resolution=cfg.RESOLUTION)
            grid_map.seed = seed # Store for reproduction logging
            
            generator = MapGenerator(obstacle_density=density, inflation_radius_m=0.2, seed=seed)
            # Use extra_paths to create a more connected "local sensing" like environment
            generator.generate(
                grid_map, plow_vehicle, cfg.START_STATE, cfg.GOAL_STATE, 
                extra_paths=cfg.EXTRA_PATHS, dead_ends=cfg.DEAD_ENDS
            )
            
            checker = CollisionChecker(cfg.COLLISION_CONFIG, vehicle, grid_map)

            # Ensure start/goal are safe (Force clear if needed)
            # The MapGenerator carves paths, but random noise might still touch the exact start/goal pixels 
            if checker.check(vehicle, cfg.START_STATE, grid_map):
                generator._clear_rectangular_area(grid_map, cfg.START_STATE, cfg.CLEAR_RADIUS)
            
            if checker.check(vehicle, cfg.GOAL_STATE, grid_map):
                generator._clear_rectangular_area(grid_map, cfg.GOAL_STATE, cfg.CLEAR_RADIUS)

            # Re-check to be sure
            if checker.check(vehicle, cfg.START_STATE, grid_map) or checker.check(vehicle, cfg.GOAL_STATE, grid_map):
                print(f"  [Skip] Trial {i} Start/Goal still blocked after clearing!")
                continue

            # --- A. RRT ---
            r_params = cfg.RRT_PARAMS
            rrt = RRTPlanner(
                vehicle, checker, 
                step_size=r_params['step_size'], 
                max_iterations=r_params['max_iterations'], 
                goal_sample_rate=r_params['goal_sample_rate'], 
                goal_threshold=r_params['goal_threshold']
            )
            debugger_rrt = PlanningDebugger()
            
            t0 = time.perf_counter()
            path_rrt = rrt.plan(cfg.START_STATE, cfg.GOAL_STATE, grid_map, debugger_rrt)
            t1 = time.perf_counter()
            
            rrt_success = False
            if path_rrt:
                rrt_success = True
                stats['RRT']['success'] += 1
                stats['RRT']['time'].append((t1 - t0) * 1000)
                stats['RRT']['nodes'].append(len(debugger_rrt.expanded_nodes))
                stats['RRT']['length'].append(calculate_path_length(path_rrt))
                
                # --- B. RRT + Smoothing (Only if RRT succeeded) ---
                smoother = GreedyShortcutSmoother(vehicle, checker, grid_map)
                t_smooth_start = time.perf_counter()
                path_smooth = smoother.smooth(path_rrt, max_iterations=cfg.SMOOTHER_PARAMS['max_iterations'])
                t_smooth_end = time.perf_counter()
                
                stats['RRT+Smooth']['success'] += 1
                # Time includes RRT planning time + smoothing time
                total_time = (t1 - t0) + (t_smooth_end - t_smooth_start)
                stats['RRT+Smooth']['time'].append(total_time * 1000)
                stats['RRT+Smooth']['nodes'].append(len(debugger_rrt.expanded_nodes)) # Same nodes metric
                stats['RRT+Smooth']['length'].append(calculate_path_length(path_smooth))
                
            else:
                save_failure_snapshot(grid_map, cfg.START_STATE, cfg.GOAL_STATE, debugger_rrt, "RRT", density, i)

            # --- C. Hybrid A* ---
            # Using slightly coarser resolutions for speed if needed, but 0.5m/5deg is standard
            h_params = cfg.HYBRID_ASTAR_PARAMS
            has = HybridAStarPlanner(
                vehicle, checker, 
                xy_resolution=h_params['xy_resolution'], 
                theta_resolution=h_params['theta_resolution'], 
                step_size=h_params['step_size'], 
                analytic_expansion_ratio=h_params['analytic_expansion_ratio']
            ) 
            debugger_has = PlanningDebugger()
            
            t0 = time.perf_counter()
            path_has = has.plan(cfg.START_STATE, cfg.GOAL_STATE, grid_map, debugger_has)
            t1 = time.perf_counter()
            
            if path_has:
                stats['HybridA*']['success'] += 1
                stats['HybridA*']['time'].append((t1 - t0) * 1000)
                stats['HybridA*']['nodes'].append(len(debugger_has.expanded_nodes) if hasattr(debugger_has, 'expanded_nodes') else 0)
                stats['HybridA*']['length'].append(calculate_path_length(path_has))
            else:
                save_failure_snapshot(grid_map, cfg.START_STATE, cfg.GOAL_STATE, debugger_has, "HybridAStar", density, i)

        # --- Aggregate Stats for Density ---
        for algo in ['RRT', 'RRT+Smooth', 'HybridA*']:
            s_data = stats[algo]
            count = len(s_data['time']) # Only sucessful ones have times
            if count == 0:
                succ_rate = 0.0
                avg_time = avg_nodes = avg_len = 0.0
                std_time = std_nodes = std_len = 0.0
            else:
                # Success rate is based on total valid trials (tried to run)
                # But here 'num_trials' includes skipped ones?
                # Actually I skipped if start/goal blocked. So denominator should be (num_trials - skipped).
                # For simplicity, assumed generated maps are valid mostly.
                # Let's just use num_trials as denominator (conservative).
                
                succ_rate = (s_data['success'] / cfg.NUM_TRIALS) * 100
                avg_time = np.mean(s_data['time'])
                std_time = np.std(s_data['time'], ddof=1) if len(s_data['time']) > 1 else 0.0
                
                avg_nodes = np.mean(s_data['nodes'])
                std_nodes = np.std(s_data['nodes'], ddof=1) if len(s_data['nodes']) > 1 else 0.0
                
                avg_len = np.mean(s_data['length'])
                std_len = np.std(s_data['length'], ddof=1) if len(s_data['length']) > 1 else 0.0
            
            # Format: "Mean ± Std"
            s_time_str = f"{avg_time:.1f}±{std_time:.1f}"
            s_len_str = f"{avg_len:.1f}±{std_len:.1f}"
            s_nodes_str = f"{avg_nodes:.1f}±{std_nodes:.1f}"
            
            res_str = f"{density:<8.2f} | {algo:<12} | {succ_rate:<6.1f} | {s_time_str:<15} | {s_len_str:<15} | {s_nodes_str:<15}"
            log_experiment(res_str)
            
            results.append({
                'Density': density, 'Algorithm': algo, 'SuccessRate': succ_rate,
                'TimeMean': avg_time, 'TimeStd': std_time,
                'NodesMean': avg_nodes, 'NodesStd': std_nodes,
                'LengthMean': avg_len, 'LengthStd': std_len
            })

    return pd.DataFrame(results)

def plot_benchmark_results(df):
    """Plot metrics with Error Bars"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    
    metrics = [
        ('SuccessRate', None, 'Success Rate (%)', 'Reliability'),
        ('TimeMean', 'TimeStd', 'Execution Time (ms)', 'Time Efficiency'),
        ('NodesMean', 'NodesStd', 'Expanded Nodes', 'Search Space'),
        ('LengthMean', 'LengthStd', 'Path Length (m)', 'Path Optimality')
    ]
    
    colors = {'RRT': 'orange', 'RRT+Smooth': 'green', 'HybridA*': 'blue'}
    markers = {'RRT': 's-', 'RRT+Smooth': '^-', 'HybridA*': 'o-'}

    for i, (metric, std_metric, ylabel, title) in enumerate(metrics):
        ax = axes_flat[i]
        
        for algo in ['RRT', 'RRT+Smooth', 'HybridA*']:
            subset = df[df['Algorithm'] == algo]
            x = subset['Density']
            y = subset[metric]
            
            if std_metric:
                yerr = subset[std_metric]
                ax.errorbar(x, y, yerr=yerr, label=algo, fmt=markers[algo], color=colors[algo], capsize=4, alpha=0.8)
            else:
                ax.plot(x, y, markers[algo], label=algo, color=colors[algo], alpha=0.8)
                
        ax.set_title(title)
        ax.set_xlabel('Obstacle Density')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(cfg.RESULTS_PLOT_PATH, dpi=150)
    
    # helper for clean path printing
    def format_path(p):
        abs_p = os.path.abspath(p)
        if len(abs_p) > 1 and abs_p[1] == ':':
            return abs_p[0].upper() + abs_p[1:]
        return abs_p

    print(f"\nSaved plots to {format_path(cfg.RESULTS_PLOT_PATH)}")
    # plt.show() # Non-blocking in agent environment

if __name__ == "__main__":
    print("=== Ackermann Planner Benchmark (RRT vs RRT+Smooth vs Hybrid A*) ===")
    df = run_experiment()
    if not df.empty:
        plot_benchmark_results(df)
        
        print("\n=== Experiment Outputs ===")
        
        # helper for clean path printing
        def format_path(p):
            abs_p = os.path.abspath(p)
            if len(abs_p) > 1 and abs_p[1] == ':':
                return abs_p[0].upper() + abs_p[1:]
            return abs_p

        print(f"Results Plot: {format_path(cfg.RESULTS_PLOT_PATH)}")
        print(f"Log File: {format_path(cfg.EXPERIMENT_LOG_PATH)}")
        
        print("\nBenchmark Complete.")
    else:
        print("No results generated.")
