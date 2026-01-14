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
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs_ackermann")
os.makedirs(LOG_DIR, exist_ok=True)

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
    filepath = os.path.join(LOG_DIR, filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"  [LOG] Failure snapshot saved: {filepath}")

def run_experiment():
    # --- 1. Experiment Parameters ---
    densities = [0.10, 0.20] # Obstacle densities to test
    num_trials = 20 # Number of trials per density
    
    # Map dimensions (Meters)
    phys_width, phys_height = 100.0, 100.0
    res = 0.5
    map_width = int(phys_width / res)
    map_height = int(phys_height / res)
    
    start_state = State(5.0, 5.0, 0.0)
    goal_state = State(90.0, 90.0, 0.0)

    # Vehicle Config
    vehicle_config = AckermannConfig(
        wheelbase=2.5, 
        max_steer_deg=40.0,  # Improved maneuverability
        width=2.0, 
        front_hang=1.0, 
        rear_hang=1.0,
        safe_margin=0.2
    )
    vehicle = AckermannVehicle(vehicle_config)

    # Map Generator Vehicle (Bulldozer)
    # Slightly larger to ensure paths exist
    plow_config = AckermannConfig(
        wheelbase=2.5, max_steer_deg=35.0, width=3.0,
        front_hang=1.2, rear_hang=1.2, safe_margin=0.5
    )
    plow_vehicle = AckermannVehicle(plow_config)

    # Collision Checker (Polygon for accuracy)
    col_config = CollisionConfig(method=CollisionMethod.POLYGON)
    
    results = []

    print(f"{'Density':<8} | {'Algo':<12} | {'Succ%':<6} | {'Time(ms)':<10} | {'Len(m)':<8} | {'Nodes':<8}")
    print("-" * 85)

    # --- 2. Loop Execution ---
    for density in densities:
        
        stats = {
            'RRT': {'success': 0, 'time': [], 'nodes': [], 'length': []},
            'RRT+Smooth': {'success': 0, 'time': [], 'nodes': [], 'length': []},
            'HybridA*': {'success': 0, 'time': [], 'nodes': [], 'length': []}
        }

        for i in range(num_trials):
            seed = 1000 + int(density * 100) + i
            grid_map = GridMap(width=map_width, height=map_height, resolution=res)
            
            generator = MapGenerator(obstacle_density=density, inflation_radius_m=0.2, seed=seed)
            # Use extra_paths to create a more connected "local sensing" like environment
            generator.generate(grid_map, plow_vehicle, start_state, goal_state, extra_paths=6, dead_ends=4)
            
            checker = CollisionChecker(col_config, vehicle, grid_map)

            # Ensure start/goal are safe (Force clear if needed)
            # The MapGenerator carves paths, but random noise might still touch the exact start/goal pixels 
            if checker.check(vehicle, start_state, grid_map):
                generator._clear_rectangular_area(grid_map, start_state, 4.0)
            
            if checker.check(vehicle, goal_state, grid_map):
                generator._clear_rectangular_area(grid_map, goal_state, 4.0)

            # Re-check to be sure
            if checker.check(vehicle, start_state, grid_map) or checker.check(vehicle, goal_state, grid_map):
                print(f"  [Skip] Trial {i} Start/Goal still blocked after clearing!")
                continue

            # --- A. RRT ---
            rrt = RRTPlanner(vehicle, checker, step_size=3.0, max_iterations=10000, goal_sample_rate=0.1, goal_threshold=2.0)
            debugger_rrt = PlanningDebugger()
            
            t0 = time.perf_counter()
            path_rrt = rrt.plan(start_state, goal_state, grid_map, debugger_rrt)
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
                path_smooth = smoother.smooth(path_rrt, max_iterations=150)
                t_smooth_end = time.perf_counter()
                
                stats['RRT+Smooth']['success'] += 1
                # Time includes RRT planning time + smoothing time
                total_time = (t1 - t0) + (t_smooth_end - t_smooth_start)
                stats['RRT+Smooth']['time'].append(total_time * 1000)
                stats['RRT+Smooth']['nodes'].append(len(debugger_rrt.expanded_nodes)) # Same nodes metric
                stats['RRT+Smooth']['length'].append(calculate_path_length(path_smooth))
                
            else:
                save_failure_snapshot(grid_map, start_state, goal_state, debugger_rrt, "RRT", density, i)

            # --- C. Hybrid A* ---
            # Using slightly coarser resolutions for speed if needed, but 0.5m/5deg is standard
            has = HybridAStarPlanner(vehicle, checker, xy_resolution=0.5, theta_resolution=np.deg2rad(5.0), 
                                     step_size=1.5, analytic_expansion_ratio=0.2) 
                                     # Improved step_size for Ackermann
            debugger_has = PlanningDebugger()
            
            t0 = time.perf_counter()
            path_has = has.plan(start_state, goal_state, grid_map, debugger_has)
            t1 = time.perf_counter()
            
            if path_has:
                stats['HybridA*']['success'] += 1
                stats['HybridA*']['time'].append((t1 - t0) * 1000)
                stats['HybridA*']['nodes'].append(len(debugger_has.expanded_nodes) if hasattr(debugger_has, 'expanded_nodes') else 0)
                stats['HybridA*']['length'].append(calculate_path_length(path_has))
            else:
                save_failure_snapshot(grid_map, start_state, goal_state, debugger_has, "HybridAStar", density, i)

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
                
                succ_rate = (s_data['success'] / num_trials) * 100
                avg_time = np.mean(s_data['time'])
                std_time = np.std(s_data['time'])
                avg_nodes = np.mean(s_data['nodes'])
                std_nodes = np.std(s_data['nodes'])
                avg_len = np.mean(s_data['length'])
                std_len = np.std(s_data['length'])
            
            print(f"{density:<8.2f} | {algo:<12} | {succ_rate:<6.1f} | {avg_time:<10.1f} | {avg_len:<8.1f} | {avg_nodes:<8.1f}")
            
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
    outfile = "benchmark_ackermann_results.png"
    plt.savefig(outfile, dpi=150)
    print(f"\nSaved plots to {outfile}")
    # plt.show() # Non-blocking in agent environment

if __name__ == "__main__":
    print("=== Ackermann Planner Benchmark (RRT vs RRT+Smooth vs Hybrid A*) ===")
    df = run_experiment()
    if not df.empty:
        plot_benchmark_results(df)
        print("\nBenchmark Complete.")
    else:
        print("No results generated.")
