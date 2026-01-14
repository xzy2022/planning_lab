import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

# --- Path Setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.ackermann import AckermannVehicle
from src.collision.checker import CollisionChecker
from src.planning.planners import HybridAStarPlanner
from src.simulation.sensor import Sensor
from src.simulation.navigator import Navigator
from src.simulation.visualizer import SimulationVisualizer
from experiments.benchmark_config import BenchmarkConfig as cfg

def ensure_log_dir(log_dir):
    os.makedirs(log_dir, exist_ok=True)

def run_perception_experiment(density=0.1, seed=42, show_plot=False):
    print(f"=== Running Perception Experiment (Density={density}, Seed={seed}) ===")
    
    # 1. Setup Environment
    # Use BenchmarkConfig for consistency
    grid_map = GridMap(width=cfg.MAP_WIDTH, height=cfg.MAP_HEIGHT, resolution=cfg.RESOLUTION)
    grid_map.seed = seed
    
    vehicle = AckermannVehicle(cfg.VEHICLE_CONFIG)
    plow_vehicle = AckermannVehicle(cfg.PLOW_CONFIG) # Used for map generation paths
    
    print("Generating Global Map...")
    generator = MapGenerator(obstacle_density=density, inflation_radius_m=0.2, seed=seed)
    # Generate map with some paths carved out to make it navigable
    generator.generate(
        grid_map, plow_vehicle, cfg.START_STATE, cfg.GOAL_STATE, 
        extra_paths=cfg.EXTRA_PATHS, dead_ends=cfg.DEAD_ENDS
    )
    
    # Ensure Start/Goal are clear (using same logic as benchmark)
    checker = CollisionChecker(cfg.COLLISION_CONFIG, vehicle, grid_map)
    if checker.check(vehicle, cfg.START_STATE, grid_map):
        print("Clearing Start Area...")
        generator._clear_rectangular_area(grid_map, cfg.START_STATE, cfg.CLEAR_RADIUS)
    if checker.check(vehicle, cfg.GOAL_STATE, grid_map):
        print("Clearing Goal Area...")
        generator._clear_rectangular_area(grid_map, cfg.GOAL_STATE, cfg.CLEAR_RADIUS)

    # Final check
    if checker.check(vehicle, cfg.START_STATE, grid_map) or checker.check(vehicle, cfg.GOAL_STATE, grid_map):
        print("CRITICAL: Start or Goal is still blocked. Experiment might fail immediately.")

    # 2. Setup Simulation Components
    print("Initializing Navigator and Sensor...")
    
    # Planner used by Navigator (Hybrid A*)
    # We use same parameters as benchmark
    h_params = cfg.HYBRID_ASTAR_PARAMS
    # Note: Navigator re-initializes planner with local map, so we pass the instance
    # But wait, Navigator takes a planner instance. 
    # The navigator.traverse() -> _replan() calls planner.plan(start, goal, local_map)
    # So the planner instance must be stateless regarding the map or we update it.
    # HybridAStarPlanner.plan() takes grid_map as arg, so it's fine.
    
    planner = HybridAStarPlanner(
        vehicle, checker, 
        xy_resolution=h_params['xy_resolution'], 
        theta_resolution=h_params['theta_resolution'], 
        step_size=h_params['step_size'], 
        analytic_expansion_ratio=h_params['analytic_expansion_ratio']
    )
    
    # Sensor with limited range (e.g., 20m)
    sensor = Sensor(sensing_radius=20.0) 
    
    navigator = Navigator(
        global_map=grid_map,
        planner=planner,
        sensor=sensor,
        start=cfg.START_STATE,
        goal=cfg.GOAL_STATE,
        vehicle=vehicle
    )
    
    # 3. Visualization Setup (Optional)
    step_callback = None
    if show_plot:
        viz = SimulationVisualizer(title=f"Perception Experiment (D={density})")
        step_callback = viz.update
    
    # 4. Run Navigation Loop
    print("Starting Navigation...")
    t0 = time.time()
    success = navigator.navigate(max_steps=500, step_callback=step_callback) # 500 decision steps
    t1 = time.time()
    
    print(f"Navigation Finished. Success: {success}, Duration: {t1-t0:.2f}s")
    print(f"Total Replans: {navigator.replan_count}, Steps: {navigator.step_count}")
    
    # 5. Save Final Visualization
    save_visualization(navigator, grid_map, success, density, seed)
    
    if show_plot:
         print("Close the plot window to finish.")
         plt.show()

def save_visualization(navigator, global_map, success, density, seed):
    log_dir = "logs/perception_experiment"
    ensure_log_dir(log_dir)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 1. Global Map (Background) - Faint
    ax.imshow(global_map.data, cmap='Greys', origin='lower', 
              extent=[0, global_map.width * global_map.resolution, 
                      0, global_map.height * global_map.resolution],
              alpha=0.3, label='Global Map')

    # 2. Local Map (What was discovered) - Overlay
    # Use a different colormap or alpha to show discovered obstacles
    # Valid data in local map: 0 (free), 1 (occupied), but we might have initialized with 0.
    # We only want to show occupied cells that were discovered.
    # Or just show the whole local map overlay?
    # Let's show discovered obstacles in Red
    
    # Create an RGB image for local map
    # We need to manually construct it because imshow overlay is tricky with 0s being transparent
    
    # Fast approach: Scatter plot for occupied cells in local map
    # Only iterate if needed.
    # Or just use imshow with MaskedArray
    local_data = navigator.local_map.data
    masked_local = np.ma.masked_where(local_data == 0, local_data) # Mask free space
    
    ax.imshow(masked_local, cmap='Reds', origin='lower',
              extent=[0, global_map.width * global_map.resolution, 
                      0, global_map.height * global_map.resolution],
              alpha=0.6, vmin=0, vmax=1)
    
    # 3. Path
    # Navigated Path (History)
    path_x = [s.x for s in navigator.navigated_path]
    path_y = [s.y for s in navigator.navigated_path]
    ax.plot(path_x, path_y, 'b.-', linewidth=2, markersize=4, label='Traversed Path')
    
    # Start / Goal
    ax.plot(cfg.START_STATE.x, cfg.START_STATE.y, 'go', markersize=10, label='Start')
    ax.plot(cfg.GOAL_STATE.x, cfg.GOAL_STATE.y, 'rx', markersize=10, label='Goal')
    
    ax.set_title(f"Perception Experiment | D={density} | Seed={seed} | Res: {success}")
    ax.legend()
    
    outfile = os.path.join(log_dir, f"perception_viz_d{density}_s{seed}_{'succ' if success else 'fail'}.png")
    plt.savefig(outfile)
    print(f"Visualization saved to: {outfile}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", type=float, default=0.1, help="Obstacle density")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--show", action="store_true", help="Show plot")
    args = parser.parse_args()
    
    run_perception_experiment(args.density, args.seed, args.show)
