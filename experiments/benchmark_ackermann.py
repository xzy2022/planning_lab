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
from src.collision.checker import CollisionChecker
from src.planning.planners import RRTPlanner, HybridAStarPlanner
from src.planning.smoother import GreedyShortcutSmoother
from src.simulation.sensor import Sensor
from src.simulation.navigator import Navigator
from src.visualization.debugger import PlanningDebugger
from experiments.benchmark_config import BenchmarkConfig as cfg

class BenchmarkRunner:
    def __init__(self):
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.mode = "dynamic" if cfg.ENABLE_DYNAMIC_PERCEPTION else "static"
        self.log_dir = os.path.join(cfg.LOG_DIR, self.timestamp)
        self.log_file = os.path.join(self.log_dir, f"benchmark_{self.mode}.txt")
        self.plot_file = os.path.join(self.log_dir, f"results_{self.mode}.png")
        
        os.makedirs(self.log_dir, exist_ok=True)
        
    def log(self, msg, to_console=True):
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(msg + "\n")
        if to_console:
            print(msg)

    def calculate_path_length(self, path):
        if not path or len(path) < 2:
            return 0.0
        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1].x - path[i].x
            dy = path[i+1].y - path[i].y
            length += math.hypot(dx, dy)
        return length

    def save_failure_snapshot(self, grid_map, start, goal, debugger, algo_name, density, trial_idx):
        if not debugger:
            return
            
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(grid_map.data, cmap='Greys', origin='lower', 
                  extent=[0, grid_map.width * grid_map.resolution, 
                          0, grid_map.height * grid_map.resolution],
                  alpha=0.5)
        ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
        ax.plot(goal.x, goal.y, 'rx', markersize=10, label='Goal')
        
        if hasattr(debugger, 'expanded_nodes') and debugger.expanded_nodes:
             # Handle different node structures depending on planner
            xs, ys = [], []
            for n in debugger.expanded_nodes:
                if hasattr(n, 'x'):
                    xs.append(n.x); ys.append(n.y)
                elif isinstance(n, (list, tuple)): # RRT might store tuples or objects
                    xs.append(n[0]); ys.append(n[1])
            if xs:
                ax.scatter(xs, ys, c='red', s=2, alpha=0.6, label='Expanded')

        ax.set_title(f"FAILURE: {algo_name} | D={density} | T={trial_idx}")
        filename = f"fail_{algo_name}_d{density}_t{trial_idx}.png"


        
        filepath = os.path.join(self.log_dir, filename)
        plt.savefig(filepath)
        plt.close(fig)
        
        # Reproduction Command
        # Mapping to unified run_experiment
        # mode arg maps to our current mode
        repro_mode = "perception" if cfg.ENABLE_DYNAMIC_PERCEPTION else "static"
        repro_cmd = f"python experiments/run_experiment.py --mode {repro_mode} --algo {algo_name} --density {density} --seed {grid_map.seed} --show"
        
        self.log(f"  [SNAPSHOT] {filename}")
        self.log(f"  [REPRODUCE] {repro_cmd}")

    def run(self, trials_override=None):
        num_trials = trials_override if trials_override else cfg.NUM_TRIALS
        
        self.log(f"=== Benchmark Runner (Mode: {self.mode}) ===")
        self.log(f"Time: {self.timestamp}")
        self.log(f"Config: Densities={cfg.DENSITIES}, Trials={num_trials}")
        self.log("-" * 60)
        
        results = []
        
        # Define Header
        base_header = f"{'Density':<8} | {'Algo':<12} | {'Succ%':<6} | {'Time(ms/s)':<12} | {'Len(m)':<12}"
        if cfg.ENABLE_DYNAMIC_PERCEPTION:
            header = base_header + f" | {'Replans':<12} | {'Steps':<12}"
        else:
            header = base_header + f" | {'Nodes':<12}"
            
        self.log(header)
        self.log("-" * len(header))

        for density in cfg.DENSITIES:
            # Algorithms to test
            # In dynamic, usually just HybridA* vs RRT. 
            # In static, we also have RRT+Smooth.
            algos = ['HybridA*', 'RRT']
            if not cfg.ENABLE_DYNAMIC_PERCEPTION:
                algos.append('RRT+Smooth')
            
            stats = {algo: {'success': 0, 'time': [], 'length': [], 
                            'nodes': [], 'replans': [], 'steps': []} 
                     for algo in algos}

            for i in range(num_trials):
                seed = cfg.RANDOM_SEED_BASE + int(density * 100) + i
                
                # Setup Map
                grid_map = GridMap(width=cfg.MAP_WIDTH, height=cfg.MAP_HEIGHT, resolution=cfg.RESOLUTION)
                grid_map.seed = seed
                vehicle = AckermannVehicle(cfg.VEHICLE_CONFIG)
                plow_vehicle = AckermannVehicle(cfg.PLOW_CONFIG)
                
                generator = MapGenerator(obstacle_density=density, inflation_radius_m=0.2, seed=seed)
                generator.generate(grid_map, plow_vehicle, cfg.START_STATE, cfg.GOAL_STATE, 
                                   extra_paths=cfg.EXTRA_PATHS, dead_ends=cfg.DEAD_ENDS)
                
                checker = CollisionChecker(cfg.COLLISION_CONFIG, vehicle, grid_map)
                
                # Clear Start/Goal
                if checker.check(vehicle, cfg.START_STATE, grid_map):
                    generator._clear_rectangular_area(grid_map, cfg.START_STATE, cfg.CLEAR_RADIUS)
                if checker.check(vehicle, cfg.GOAL_STATE, grid_map):
                    generator._clear_rectangular_area(grid_map, cfg.GOAL_STATE, cfg.CLEAR_RADIUS)
                
                if checker.check(vehicle, cfg.START_STATE, grid_map) or checker.check(vehicle, cfg.GOAL_STATE, grid_map):
                    self.log(f"  [Skip] Trial {i} Start/Goal blocked.")
                    continue

                # Run Algorithms
                for algo_name in algos:
                    # RRT+Smooth is special case of RRT
                    current_algo_runner = algo_name
                    if algo_name == 'RRT+Smooth':
                         # We've already run RRT in this loop iteration logically? 
                         # Actually for fairness we should re-run or cache.
                         # benchmark_ackermann runs RRT then Smooths. 
                         # Let's handle it simply: RRT+Smooth does a fresh RRT then smooth.
                         current_algo_runner = 'RRT' 
                         
                    planner = self._create_planner(current_algo_runner, vehicle, checker)
                    
                    if cfg.ENABLE_DYNAMIC_PERCEPTION:
                        self._run_dynamic(planner, algo_name, grid_map, vehicle, stats, density, i, seed)
                    else:
                        self._run_static(planner, algo_name, grid_map, vehicle, checker, stats, density, i)

            # --- Aggregate Stats for Density ---
            for algo_name in algos:
                self._log_stats(stats[algo_name], algo_name, density, num_trials, results)

        # Convert to DataFrame and Plot
        df = pd.DataFrame(results)
        if not df.empty:
            self._plot_results(df)
            self.log(f"\nPlots saved to {self.plot_file}")
            
        print("Benchmark Complete.")

    def _create_planner(self, algo_name, vehicle, checker):
        if algo_name == 'RRT':
            r_params = cfg.RRT_PARAMS
            return RRTPlanner(vehicle, checker, **r_params)
        elif algo_name == 'HybridA*':
            h_params = cfg.HYBRID_ASTAR_PARAMS
            return HybridAStarPlanner(vehicle, checker, **h_params)
        return None

    def _run_dynamic(self, planner, algo_name, grid_map, vehicle, stats, density, trial_idx, seed):
        sensor = Sensor(sensing_radius=20.0)
        navigator = Navigator(grid_map, planner, sensor, cfg.START_STATE, cfg.GOAL_STATE, vehicle)
        
        t0 = time.time()
        success = navigator.navigate(max_steps=500)
        t1 = time.time()
        
        if success:
            stats[algo_name]['success'] += 1
            stats[algo_name]['time'].append(t1 - t0) # Seconds
            stats[algo_name]['length'].append(self.calculate_path_length(navigator.navigated_path))
            stats[algo_name]['replans'].append(navigator.replan_count)
            stats[algo_name]['steps'].append(navigator.step_count)
        else:
            # Fake a debugger-like object or just pass None for now, snapshotting for dynamic is harder
            # because we need the final local map state.
            # But wait, benchmark_perception used 'navigator' state which isn't a debugger.
            # We can just log the reproduction command.
            repro_cmd = f"python experiments/run_experiment.py --mode perception --algo {algo_name} --density {density} --seed {seed} --show"
            # We only verify one failed case usually
            if stats[algo_name]['success'] == 0 and len(stats[algo_name]['time']) == 0: 
                 # Just log first failure of batch to avoid spam? or all? 
                 # Let's log formatted failure
                 pass
            self.log(f"  [FAILURE] {algo_name} D={density} T={trial_idx} -> {repro_cmd}")

    def _run_static(self, planner, algo_name, grid_map, vehicle, checker, stats, density, trial_idx):
        debugger = PlanningDebugger()
        t0 = time.perf_counter()
        path = planner.plan(cfg.START_STATE, cfg.GOAL_STATE, grid_map, debugger)
        t1 = time.perf_counter()
        
        if algo_name == 'RRT+Smooth':
            if path:
                smoother = GreedyShortcutSmoother(vehicle, checker, grid_map)
                t_s = time.perf_counter()
                path = smoother.smooth(path, max_iterations=cfg.SMOOTHER_PARAMS['max_iterations'])
                t_e = time.perf_counter()
                # Total time
                stats[algo_name]['time'].append((t1 - t0 + t_e - t_s) * 1000) # ms
            else:
                # Failed RRT means Failed Smooth
                pass 
        else:
             if path:
                 stats[algo_name]['time'].append((t1 - t0) * 1000) # ms

        if path:
            stats[algo_name]['success'] += 1
            stats[algo_name]['length'].append(self.calculate_path_length(path))
            stats[algo_name]['nodes'].append(len(debugger.expanded_nodes))
        else:
            self.save_failure_snapshot(grid_map, cfg.START_STATE, cfg.GOAL_STATE, debugger, algo_name, density, trial_idx)

    def _log_stats(self, s_data, algo_name, density, num_trials, results):
        if s_data['success'] > 0:
            succ_rate = (s_data['success'] / num_trials) * 100
            
            # Helper to get mean/std
            def get_ms(key):
                vals = s_data[key]
                if not vals: return 0.0, 0.0
                return np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

            m_time, s_time = get_ms('time')
            m_len, s_len = get_ms('length')
            
            row_str = f"{density:<8.2f} | {algo_name:<12} | {succ_rate:<6.1f} | {m_time:.1f}±{s_time:.1f}      | {m_len:.1f}±{s_len:.1f}     "
            
            stats_dict = {
                'Density': density, 'Algorithm': algo_name, 'SuccessRate': succ_rate,
                'TimeMean': m_time, 'TimeStd': s_time,
                'LengthMean': m_len, 'LengthStd': s_len
            }

            if cfg.ENABLE_DYNAMIC_PERCEPTION:
                 m_rep, s_rep = get_ms('replans')
                 m_step, s_step = get_ms('steps')
                 row_str += f" | {m_rep:.1f}±{s_rep:.1f}      | {m_step:.1f}±{s_step:.1f}"
                 stats_dict.update({'ReplansMean': m_rep, 'ReplansStd': s_rep, 'StepsMean': m_step, 'StepsStd': s_step})
            else:
                 m_nodes, s_nodes = get_ms('nodes')
                 row_str += f" | {m_nodes:.1f}±{s_nodes:.1f}"
                 stats_dict.update({'NodesMean': m_nodes, 'NodesStd': s_nodes})

            self.log(row_str)
            results.append(stats_dict)
        else:
            self.log(f"{density:<8.2f} | {algo_name:<12} | {0.0:<6.1f} | -            | -            | -")

    def _plot_results(self, df):
        # Determine metrics to plot
        if cfg.ENABLE_DYNAMIC_PERCEPTION:
            metrics = [
                ('SuccessRate', None, 'Success Rate (%)'),
                ('TimeMean', 'TimeStd', 'Time (s)'),
                ('ReplansMean', 'ReplansStd', 'Replans'),
                ('StepsMean', 'StepsStd', 'Steps')
            ]
        else:
             metrics = [
                ('SuccessRate', None, 'Success Rate (%)'),
                ('TimeMean', 'TimeStd', 'Time (ms)'),
                ('NodesMean', 'NodesStd', 'Expanded Nodes'),
                ('LengthMean', 'LengthStd', 'Path Length (m)')
            ]
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        colors = {'RRT': 'orange', 'RRT+Smooth': 'green', 'HybridA*': 'blue'}
        markers = {'RRT': 's-', 'RRT+Smooth': '^-', 'HybridA*': 'o-'}
        
        for i, (metric, std_metric, ylabel) in enumerate(metrics):
            ax = axes[i]
            for algo in df['Algorithm'].unique():
                subset = df[df['Algorithm'] == algo]
                if std_metric and metric in subset:
                    ax.errorbar(subset['Density'], subset[metric], yerr=subset[std_metric], 
                                label=algo, fmt=markers.get(algo, 'o-'), color=colors.get(algo, 'gray'), capsize=4)
                elif metric in subset:
                    ax.plot(subset['Density'], subset[metric], markers.get(algo, 'o-'), 
                            label=algo, color=colors.get(algo, 'gray'))
            
            ax.set_title(ylabel)
            ax.set_xlabel('Density')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.suptitle(f"Benchmark Results ({self.mode})", fontsize=16)
        plt.subplots_adjust(top=0.92)
        plt.savefig(self.plot_file)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None, help="Override number of trials per density")
    args = parser.parse_args()
    
    runner = BenchmarkRunner()
    runner.run(trials_override=args.trials)