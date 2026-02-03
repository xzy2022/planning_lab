import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

# --- Path Setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.ackermann import AckermannVehicle
from src.collision.checker import CollisionChecker
from src.planning.planners import HybridAStarPlanner, RRTPlanner
from src.simulation.sensor import Sensor
from src.simulation.navigator import Navigator
from src.simulation.visualizer import SimulationVisualizer
from src.visualization.observers import DebugObserver
from experiments.benchmark_config import BenchmarkConfig as cfg

# Fixed temp file for replay data
TEMP_REPLAY_FILE = "temp_replay_data.pkl"

def ensure_log_dir(log_dir):
    os.makedirs(log_dir, exist_ok=True)

def run_experiment(mode="perception", density=0.1, seed=42, algo="HybridA*", show_plot=False, skip=1):
    print(f"=== Running Experiment (Mode={mode}, Algo={algo}, Density={density}, Seed={seed}) ===")
    
    # 1. Setup Environment
    grid_map = GridMap(width=cfg.MAP_WIDTH, height=cfg.MAP_HEIGHT, resolution=cfg.RESOLUTION)
    grid_map.seed = seed
    
    vehicle = AckermannVehicle(cfg.VEHICLE_CONFIG)
    plow_vehicle = AckermannVehicle(cfg.PLOW_CONFIG)
    
    print("Generating Global Map...")
    generator = MapGenerator(obstacle_density=density, inflation_radius_m=0.2, seed=seed)
    generator.generate(
        grid_map, plow_vehicle, cfg.START_STATE, cfg.GOAL_STATE, 
        extra_paths=cfg.EXTRA_PATHS, dead_ends=cfg.DEAD_ENDS
    )
    
    checker = CollisionChecker(cfg.COLLISION_CONFIG, vehicle, grid_map)
    
    # Ensure Start/Goal are clear
    if checker.check(vehicle, cfg.START_STATE, grid_map):
        print("Clearing Start Area...")
        generator._clear_rectangular_area(grid_map, cfg.START_STATE, cfg.CLEAR_RADIUS)
    if checker.check(vehicle, cfg.GOAL_STATE, grid_map):
        print("Clearing Goal Area...")
        generator._clear_rectangular_area(grid_map, cfg.GOAL_STATE, cfg.CLEAR_RADIUS)

    if checker.check(vehicle, cfg.START_STATE, grid_map) or checker.check(vehicle, cfg.GOAL_STATE, grid_map):
        print("CRITICAL: Start or Goal is still blocked. Experiment might fail.")

    # 2. Setup Planner
    planner = None
    if algo == "HybridA*":
        h_params = cfg.HYBRID_ASTAR_PARAMS
        planner = HybridAStarPlanner(
            vehicle, checker, 
            xy_resolution=h_params['xy_resolution'], 
            theta_resolution=h_params['theta_resolution'], 
            step_size=h_params['step_size'], 
            analytic_expansion_ratio=h_params['analytic_expansion_ratio']
        )
    elif algo == "RRT":
        r_params = cfg.RRT_PARAMS
        planner = RRTPlanner(
            vehicle, checker,
            step_size=r_params['step_size'],
            max_iterations=r_params['max_iterations'],
            goal_sample_rate=r_params['goal_sample_rate'],
            goal_threshold=r_params['goal_threshold']
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # 3. Execution based on Mode
    if mode == "static":
        run_static_mode(planner, grid_map, algo, density, seed)
    elif mode == "perception":
        run_perception_mode(planner, grid_map, vehicle, density, seed, show_plot, skip)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def run_static_mode(planner, grid_map, algo, density, seed):
    print("--- Static Debug Mode ---")
    observer = DebugObserver(log_dir="logs/planning_debug")
    print(f"Debug Log initialized: {observer.logger.handlers[0].baseFilename}")
    
    print("Starting planner...")
    t0 = time.perf_counter()
    path = planner.plan(cfg.START_STATE, cfg.GOAL_STATE, grid_map, debugger=observer)
    t1 = time.perf_counter()
    
    duration_ms = (t1 - t0) * 1000
    success = len(path) > 0
    print(f"Planning Finished. Success: {success}, Time: {duration_ms:.2f} ms")
    
    # Save Visualization
    save_static_viz(grid_map, cfg.START_STATE, cfg.GOAL_STATE, path, observer, algo, density, seed, success)

def run_perception_mode(planner, grid_map, vehicle, density, seed, show_plot, skip):
    print("--- Dynamic Perception Mode ---")
    sensor = Sensor(sensing_radius=20.0)
    
    navigator = Navigator(
        global_map=grid_map,
        planner=planner,
        sensor=sensor,
        start=cfg.START_STATE,
        goal=cfg.GOAL_STATE,
        vehicle=vehicle
    )
    
    # Recorder Logic
    frames = []
    
    def recorder_callback(nav: Navigator):
        # Capture critical data for visualization
        frame = {
            'local_map_data': np.copy(nav.local_map.data), # Copy to avoid reference issues
            'vehicle_x': nav.current_state.x,
            'vehicle_y': nav.current_state.y,
            'path_x': [s.x for s in nav.navigated_path],
            'path_y': [s.y for s in nav.navigated_path],
            'step': nav.step_count,
            'replans': nav.replan_count
        }
        frames.append(frame)

    print("Starting Navigation (Recording)...")
    t0 = time.time()
    # If show_plot is enabled, we use the recorder callback
    callback = recorder_callback if show_plot else None
    
    success = navigator.navigate(max_steps=cfg.MAX_STEPS, step_callback=callback)
    t1 = time.time()
    
    print(f"Navigation Finished. Success: {success}, Duration: {t1-t0:.2f}s")
    print(f"Total Replans: {navigator.replan_count}, Steps: {navigator.step_count}")
    
    # Save result image FIRST (Backend AGG to avoid window popups if possible)
    # But saving first is good.
    save_perception_viz(navigator, grid_map, success, density, seed)
    
    # Replay Phase
    if show_plot and frames:
        print(f"Captured {len(frames)} frames. Starting Replay (Skip={skip})...")
        
        # Save to fixed temp file (overwrite if exists)
        with open(TEMP_REPLAY_FILE, 'wb') as f:
            pickle.dump(frames, f)
        
        try:
            # Replay
            viz = SimulationVisualizer(title=f"Replay: D={density}")
            # Initialize with full navigator state context
            viz._init_plot(navigator) 
            
            with open(TEMP_REPLAY_FILE, 'rb') as f:
                loaded_frames = pickle.load(f)
            
            # Use small constant interval for max speed rendering
            fast_interval = 0.001 
            
            # Iterate with skip
            for i in range(0, len(loaded_frames), skip):
                frame = loaded_frames[i]
                viz.update_from_state(frame, pause_interval=fast_interval)
            
            # Ensure final frame is shown if missed
            if (len(loaded_frames) - 1) % skip != 0:
                 viz.update_from_state(loaded_frames[-1], pause_interval=fast_interval)

            print("Replay Finished. Window is frozen. Close window to clean up and exit.")
            plt.show() # Blocks until window is closed
            
        finally:
            # Cleanup
            if os.path.exists(TEMP_REPLAY_FILE):
                try:
                    os.remove(TEMP_REPLAY_FILE)
                    print(f"Cleanup: Removed {TEMP_REPLAY_FILE}")
                except Exception as e:
                    print(f"Warning: Failed to remove temp file: {e}")

def save_static_viz(grid_map, start, goal, path, observer, algo, density, seed, success):
    ensure_log_dir("logs/planning_debug")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    ax.imshow(grid_map.data, cmap='Greys', origin='lower', 
              extent=[0, grid_map.width * grid_map.resolution, 
                      0, grid_map.height * grid_map.resolution],
              alpha=0.5)
    
    if observer.expanded_nodes:
        xs, ys = [], []
        for n in observer.expanded_nodes:
            if hasattr(n, 'x'):
                xs.append(n.x); ys.append(n.y)
            elif isinstance(n, (list, tuple)):
                xs.append(n[0]); ys.append(n[1])
        ax.scatter(xs, ys, c='orange', s=2, alpha=0.5, label='Expanded')

    if hasattr(observer, 'edges') and observer.edges:
        for (s, e) in observer.edges:
             ax.plot([s.x, e.x], [s.y, e.y], 'y-', linewidth=0.5, alpha=0.3)

    if path:
        px = [p.x for p in path]; py = [p.y for p in path]
        ax.plot(px, py, 'b-', linewidth=2, label='Path')
        
    ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
    ax.plot(goal.x, goal.y, 'rx', markersize=10, label='Goal')
    
    ax.set_title(f"DEBUG: {algo} | D={density} | Seed={seed} | {'SUCCESS' if success else 'FAIL'}")
    ax.legend()
    
    outfile = f"logs/planning_debug/debug_viz_{algo}_{seed}.png"
    plt.savefig(outfile)
    plt.close(fig) # Ensure closure
    print(f"Visualization saved to: {outfile}")

def save_perception_viz(navigator, global_map, success, density, seed):
    log_dir = "logs/perception_experiment"
    ensure_log_dir(log_dir)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    ax.imshow(global_map.data, cmap='Greys', origin='lower', 
              extent=[0, global_map.width * global_map.resolution, 
                      0, global_map.height * global_map.resolution],
              alpha=0.3, label='Global Map')

    local_data = navigator.local_map.data
    masked_local = np.ma.masked_where(local_data == 0, local_data)
    
    ax.imshow(masked_local, cmap='Reds', origin='lower',
              extent=[0, global_map.width * global_map.resolution, 
                      0, global_map.height * global_map.resolution],
              alpha=0.6, vmin=0, vmax=1)
    
    path_x = [s.x for s in navigator.navigated_path]
    path_y = [s.y for s in navigator.navigated_path]
    ax.plot(path_x, path_y, 'b.-', linewidth=2, markersize=4, label='Traversed Path')
    
    ax.plot(cfg.START_STATE.x, cfg.START_STATE.y, 'go', markersize=10, label='Start')
    ax.plot(cfg.GOAL_STATE.x, cfg.GOAL_STATE.y, 'rx', markersize=10, label='Goal')
    
    ax.set_title(f"Perception Experiment | D={density} | Seed={seed} | Res: {success}")
    ax.legend()
    
    outfile = os.path.join(log_dir, f"perception_viz_d{density}_s{seed}_{'succ' if success else 'fail'}.png")
    plt.savefig(outfile)
    plt.close(fig) # Ensure closure to avoid extra windows
    print(f"Visualization saved to: {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="perception", choices=["perception", "static"], help="Experiment mode")
    parser.add_argument("--density", type=float, default=0.1, help="Obstacle density")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--algo", type=str, default="HybridA*", choices=["HybridA*", "RRT"], help="Algorithm")
    parser.add_argument("--show", action="store_true", help="Show plot")
    parser.add_argument("--skip", type=int, default=10, help="Animation skip steps")
    args = parser.parse_args()
    
    run_experiment(args.mode, args.density, args.seed, args.algo, args.show, args.skip)
