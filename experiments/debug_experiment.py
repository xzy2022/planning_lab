import argparse
import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# --- Path Setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.ackermann import AckermannVehicle
from src.collision.checker import CollisionChecker
from src.planning.planners import RRTPlanner, HybridAStarPlanner
from src.visualization.observers import DebugObserver
from experiments.benchmark_config import BenchmarkConfig as cfg

def run_debug_session(algo_name, density, seed):
    print(f"=== Debug Session: {algo_name} | Density: {density} | Seed: {seed} ===")
    
    # 1. Setup Environment (Exactly like Benchmark)
    grid_map = GridMap(width=cfg.MAP_WIDTH, height=cfg.MAP_HEIGHT, resolution=cfg.RESOLUTION)
    grid_map.seed = seed
    
    vehicle = AckermannVehicle(cfg.VEHICLE_CONFIG)
    plow_vehicle = AckermannVehicle(cfg.PLOW_CONFIG)
    
    print("Generating map...")
    generator = MapGenerator(obstacle_density=density, inflation_radius_m=0.2, seed=seed)
    generator.generate(
        grid_map, plow_vehicle, cfg.START_STATE, cfg.GOAL_STATE, 
        extra_paths=cfg.EXTRA_PATHS, dead_ends=cfg.DEAD_ENDS
    )
    
    checker = CollisionChecker(cfg.COLLISION_CONFIG, vehicle, grid_map)
    
    # Start/Goal Clearing Logic
    if checker.check(vehicle, cfg.START_STATE, grid_map):
        print("Start blocked, clearing...")
        generator._clear_rectangular_area(grid_map, cfg.START_STATE, cfg.CLEAR_RADIUS)
    
    if checker.check(vehicle, cfg.GOAL_STATE, grid_map):
        print("Goal blocked, clearing...")
        generator._clear_rectangular_area(grid_map, cfg.GOAL_STATE, cfg.CLEAR_RADIUS)

    if checker.check(vehicle, cfg.START_STATE, grid_map) or checker.check(vehicle, cfg.GOAL_STATE, grid_map):
        print("CRITICAL: Start/Goal still blocked after clearing! This might be the cause of failure.")
    
    # 2. Setup Planner & Observer (Mode 3)
    observer = DebugObserver(log_dir="logs/planning_debug")
    print(f"Debug Log initialized: {observer.logger.handlers[0].baseFilename}")
    
    planner = None
    if algo_name == "RRT":
        r_params = cfg.RRT_PARAMS
        planner = RRTPlanner(
            vehicle, checker, 
            step_size=r_params['step_size'], 
            max_iterations=r_params['max_iterations'], 
            goal_sample_rate=r_params['goal_sample_rate'], 
            goal_threshold=r_params['goal_threshold']
        )
    elif algo_name == "HybridAStar":
        h_params = cfg.HYBRID_ASTAR_PARAMS
        planner = HybridAStarPlanner(
            vehicle, checker, 
            xy_resolution=h_params['xy_resolution'], 
            theta_resolution=h_params['theta_resolution'], 
            step_size=h_params['step_size'], 
            analytic_expansion_ratio=h_params['analytic_expansion_ratio']
        )
    else:
        print(f"Unknown algorithm: {algo_name}")
        return

    # 3. Execute
    print("Starting planner...")
    t0 = time.perf_counter()
    path = planner.plan(cfg.START_STATE, cfg.GOAL_STATE, grid_map, debugger=observer)
    t1 = time.perf_counter()
    
    duration_ms = (t1 - t0) * 1000
    success = len(path) > 0
    print(f"Planning Finished. Success: {success}, Time: {duration_ms:.2f} ms, Nodes: {len(observer.expanded_nodes)}")
    
    # 4. Save Visualization
    save_viz(grid_map, cfg.START_STATE, cfg.GOAL_STATE, path, observer, algo_name, density, seed, success)

def save_viz(grid_map, start, goal, path, observer, algo, density, seed, success):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Background
    ax.imshow(grid_map.data, cmap='Greys', origin='lower', 
              extent=[0, grid_map.width * grid_map.resolution, 
                      0, grid_map.height * grid_map.resolution],
              alpha=0.5)
    
    # Expanded Nodes
    if observer.expanded_nodes:
        # Extract x,y from nodes depending on type
        xs, ys = [], []
        for n in observer.expanded_nodes:
            if hasattr(n, 'x'):
                xs.append(n.x)
                ys.append(n.y)
            elif isinstance(n, (list, tuple)):
                xs.append(n[0])
                ys.append(n[1])
                
        ax.scatter(xs, ys, c='orange', s=2, alpha=0.5, label='Expanded')

    # Edges (RRT)
    if hasattr(observer, 'edges') and observer.edges:
        for (s, e) in observer.edges:
             ax.plot([s.x, e.x], [s.y, e.y], 'y-', linewidth=0.5, alpha=0.3)

    # Path
    if path:
        px = [p.x for p in path]
        py = [p.y for p in path]
        ax.plot(px, py, 'b-', linewidth=2, label='Path')
        
    ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
    ax.plot(goal.x, goal.y, 'rx', markersize=10, label='Goal')
    
    ax.set_title(f"DEBUG: {algo} | D={density} | Seed={seed} | {'SUCCESS' if success else 'FAIL'}")
    ax.legend()
    
    outfile = f"logs/planning_debug/debug_viz_{algo}_{seed}.png"
    plt.savefig(outfile)
    print(f"Visualization saved to: {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug specific planning experiment")
    parser.add_argument("--algo", type=str, required=True, choices=["RRT", "HybridAStar"], help="Algorithm name")
    parser.add_argument("--density", type=float, required=True, help="Obstacle density")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    
    args = parser.parse_args()
    
    run_debug_session(args.algo, args.density, args.seed)
