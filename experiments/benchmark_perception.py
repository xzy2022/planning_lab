import sys
import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# --- Path Setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.ackermann import AckermannVehicle
from src.types import State
from src.collision.checker import CollisionChecker
from src.planning.planners import HybridAStarPlanner, RRTPlanner
from src.simulation.sensor import Sensor
from src.simulation.navigator import Navigator
from experiments.benchmark_config import BenchmarkConfig as cfg

# Override config for logging if needed, or use specific log file
PERCEPTION_LOG_PATH = os.path.join(cfg.LOG_DIR, "benchmark_perception_log.txt")
PERCEPTION_PLOT_PATH = os.path.join(cfg.LOG_DIR, "benchmark_perception_results.png")

def log_experiment(msg, to_console=True):
    with open(PERCEPTION_LOG_PATH, "a") as f:
        f.write(msg + "\n")
    if to_console:
        print(msg)

def calculate_path_length(path):
    if not path or len(path) < 2:
        return 0.0
    length = 0.0
    for i in range(len(path) - 1):
        dx = path[i+1].x - path[i].x
        dy = path[i+1].y - path[i].y
        length += math.hypot(dx, dy)
    return length

def run_benchmark(num_trials_override=None):
    # Setup Logging
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    if os.path.exists(PERCEPTION_LOG_PATH):
        os.remove(PERCEPTION_LOG_PATH)
        
    log_experiment("=== Perception Benchmark (Local Sensing + Replanning) ===")
    log_experiment(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    num_trials = num_trials_override if num_trials_override else cfg.NUM_TRIALS
    
    results = []
    
    header = f"{'Density':<8} | {'Algo':<12} | {'Succ%':<6} | {'Time(s)':<15} | {'Len(m)':<15} | {'Replans':<15} | {'Steps':<15}"
    log_experiment(header)
    log_experiment("-" * 110)
    
    # Iterate Densities
    for density in cfg.DENSITIES:
        
        stats = {
            'RRT': {'success': 0, 'time': [], 'length': [], 'replans': [], 'steps': []},
            'HybridA*': {'success': 0, 'time': [], 'length': [], 'replans': [], 'steps': []}
        }
        
        for i in range(num_trials):
            seed = cfg.RANDOM_SEED_BASE + int(density * 100) + i
            
            # --- 1. Environment ---
            grid_map = GridMap(width=cfg.MAP_WIDTH, height=cfg.MAP_HEIGHT, resolution=cfg.RESOLUTION)
            grid_map.seed = seed
            
            vehicle = AckermannVehicle(cfg.VEHICLE_CONFIG)
            plow_vehicle = AckermannVehicle(cfg.PLOW_CONFIG)
            
            generator = MapGenerator(obstacle_density=density, inflation_radius_m=0.2, seed=seed)
            generator.generate(
                grid_map, plow_vehicle, cfg.START_STATE, cfg.GOAL_STATE, 
                extra_paths=cfg.EXTRA_PATHS, dead_ends=cfg.DEAD_ENDS
            )
            
            checker = CollisionChecker(cfg.COLLISION_CONFIG, vehicle, grid_map)
            # Ensure Start/Goal are clear
            if checker.check(vehicle, cfg.START_STATE, grid_map):
                generator._clear_rectangular_area(grid_map, cfg.START_STATE, cfg.CLEAR_RADIUS)
            if checker.check(vehicle, cfg.GOAL_STATE, grid_map):
                generator._clear_rectangular_area(grid_map, cfg.GOAL_STATE, cfg.CLEAR_RADIUS)
                
            if checker.check(vehicle, cfg.START_STATE, grid_map) or checker.check(vehicle, cfg.GOAL_STATE, grid_map):
                print(f"  [Skip] Trial {i} Start/Goal blocked.")
                continue
            
            # --- Iterate Algorithms ---
            for algo_name in ['RRT', 'HybridA*']:
                planner = None
                if algo_name == 'RRT':
                    r_params = cfg.RRT_PARAMS
                    planner = RRTPlanner(
                        vehicle, checker, 
                        step_size=r_params['step_size'], 
                        max_iterations=r_params['max_iterations'], 
                        goal_sample_rate=r_params['goal_sample_rate'], 
                        goal_threshold=r_params['goal_threshold']
                    )
                elif algo_name == 'HybridA*':
                    h_params = cfg.HYBRID_ASTAR_PARAMS
                    planner = HybridAStarPlanner(
                        vehicle, checker, 
                        xy_resolution=h_params['xy_resolution'], 
                        theta_resolution=h_params['theta_resolution'], 
                        step_size=h_params['step_size'], 
                        analytic_expansion_ratio=h_params['analytic_expansion_ratio']
                    )
                
                sensor = Sensor(sensing_radius=20.0)
                
                navigator = Navigator(
                    global_map=grid_map,
                    planner=planner,
                    sensor=sensor,
                    start=cfg.START_STATE,
                    goal=cfg.GOAL_STATE,
                    vehicle=vehicle
                )
                
                # --- Execute ---
                t0 = time.time()
                success = navigator.navigate(max_steps=500)
                t1 = time.time()
                duration = t1 - t0
                
                if success:
                    stats[algo_name]['success'] += 1
                    stats[algo_name]['time'].append(duration)
                    stats[algo_name]['length'].append(calculate_path_length(navigator.navigated_path))
                    stats[algo_name]['replans'].append(navigator.replan_count)
                    stats[algo_name]['steps'].append(navigator.step_count)
                else:
                    # Log failure reproduction command
                    reproduce_cmd = f"python experiments/run_perception_experiment.py --density {density} --seed {seed} --algo {algo_name} --show"
                    fail_msg = f"[FAILURE] {algo_name} at Density {density}, Seed {seed}. \nStart: {cfg.START_STATE}, Goal: {cfg.GOAL_STATE}\nReproduction: {reproduce_cmd}"
                    log_experiment(fail_msg)
        
        # --- Aggregate and Log ---
        for algo_name in ['RRT', 'HybridA*']:
            s_data = stats[algo_name]
            if len(s_data['time']) > 0:
                succ_rate = (s_data['success'] / num_trials) * 100
                avg_time = np.mean(s_data['time'])
                std_time = np.std(s_data['time'], ddof=1) if len(s_data['time']) > 1 else 0.0
                
                avg_len = np.mean(s_data['length'])
                std_len = np.std(s_data['length'], ddof=1) if len(s_data['length']) > 1 else 0.0
                
                avg_replans = np.mean(s_data['replans'])
                std_replans = np.std(s_data['replans'], ddof=1) if len(s_data['replans']) > 1 else 0.0
                
                avg_steps = np.mean(s_data['steps'])
                std_steps = np.std(s_data['steps'], ddof=1) if len(s_data['steps']) > 1 else 0.0
            else:
                succ_rate = 0.0
                avg_time = avg_len = avg_replans = avg_steps = 0.0
                std_time = std_len = std_replans = std_steps = 0.0
                
            s_time = f"{avg_time:.2f}±{std_time:.2f}"
            s_len = f"{avg_len:.1f}±{std_len:.1f}"
            s_replans = f"{avg_replans:.1f}±{std_replans:.1f}"
            s_steps = f"{avg_steps:.1f}±{std_steps:.1f}"
            
            log_experiment(f"{density:<8.2f} | {algo_name:<12} | {succ_rate:<6.1f} | {s_time:<15} | {s_len:<15} | {s_replans:<15} | {s_steps:<15}")
            
            results.append({
                'Density': density,
                'Algorithm': algo_name,
                'SuccessRate': succ_rate,
                'TimeMean': avg_time, 'TimeStd': std_time,
                'LengthMean': avg_len, 'LengthStd': std_len,
                'ReplansMean': avg_replans, 'ReplansStd': std_replans,
                'StepsMean': avg_steps, 'StepsStd': std_steps
            })
        
    return pd.DataFrame(results)

def plot_results(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 1. Success Rate
    ax = axes[0]
    for algo in df['Algorithm'].unique():
        subset = df[df['Algorithm'] == algo]
        ax.plot(subset['Density'], subset['SuccessRate'], 'o-', label=algo)
    ax.set_title('Success Rate (%)')
    ax.set_xlabel('Density')
    ax.set_ylabel('Success %')
    ax.legend()
    ax.grid(True)
    
    # 2. Time
    ax = axes[1]
    for algo in df['Algorithm'].unique():
        subset = df[df['Algorithm'] == algo]
        ax.errorbar(subset['Density'], subset['TimeMean'], yerr=subset['TimeStd'], fmt='o-', label=algo, capsize=4)
    ax.set_title('Average Navigation Time (s)')
    ax.set_xlabel('Density')
    ax.set_ylabel('Time (s)')
    ax.legend()
    ax.grid(True)
    
    # 3. Replans
    ax = axes[2]
    for algo in df['Algorithm'].unique():
        subset = df[df['Algorithm'] == algo]
        ax.errorbar(subset['Density'], subset['ReplansMean'], yerr=subset['ReplansStd'], fmt='o-', label=algo, capsize=4)
    ax.set_title('Average Replan Count')
    ax.set_xlabel('Density')
    ax.set_ylabel('Replans')
    ax.legend()
    ax.grid(True)
    
    # 4. Steps
    ax = axes[3]
    for algo in df['Algorithm'].unique():
        subset = df[df['Algorithm'] == algo]
        ax.errorbar(subset['Density'], subset['StepsMean'], yerr=subset['StepsStd'], fmt='o-', label=algo, capsize=4)
    ax.set_title('Average Steps To Goal')
    ax.set_xlabel('Density')
    ax.set_ylabel('Steps')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(PERCEPTION_PLOT_PATH)
    print(f"\nPlots saved to {PERCEPTION_PLOT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None, help="Override number of trials per density")
    args = parser.parse_args()
    
    df = run_benchmark(num_trials_override=args.trials)
    if not df.empty:
        plot_results(df)
        print("\nBenchmark Complete.")
