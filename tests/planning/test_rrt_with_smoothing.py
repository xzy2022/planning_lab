
import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# --- Path Setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.ackermann import AckermannVehicle
from src.vehicles.config import AckermannConfig
from src.types import State
from src.collision.checker import CollisionChecker, CollisionConfig, CollisionMethod
from src.planning.planners import RRTPlanner 
from src.planning.smoother import GreedyShortcutSmoother # Import Smoother
from src.visualization.debugger import PlanningDebugger

def test_rrt_with_smoothing():
    print("=== Ackermann RRT + Smoothing Test ===")

    # 1. Map Initialization
    width, height, res = 200, 200, 0.5
    grid_map = GridMap(width=width, height=height, resolution=res)
    
    # 2. Vehicle Configuration
    vehicle_config = AckermannConfig(
        wheelbase=2.5, 
        max_steer_deg=35.0, 
        width=2.0, 
        front_hang=0.9, 
        rear_hang=0.9,
        safe_margin=0.2
    )
    vehicle = AckermannVehicle(vehicle_config)
    
    # 3. Bulldozer for Map Generation
    plow_config = AckermannConfig(
        wheelbase=2.5, 
        max_steer_deg=35.0, 
        width=2.5,
        front_hang=1.2,
        rear_hang=1.2,
        safe_margin=0.5
    )
    plow_vehicle = AckermannVehicle(plow_config)

    print("Generating map... (Bulldozer Mode)")
    generator = MapGenerator(
        obstacle_density=0.20, 
        inflation_radius_m=0.1, 
        num_waypoints=3,
        seed=1234 
    )
    start_state = State(10.0, 10.0, 0.0)
    goal_state = State(90.0, 90.0, 0.0)
    
    generator.generate(grid_map, plow_vehicle, start_state, goal_state, extra_paths=4, dead_ends=2)

    # 4. Collision Checker
    col_config = CollisionConfig(method=CollisionMethod.POLYGON)
    collision_checker = CollisionChecker(col_config, vehicle, grid_map)

    # Ensure start is safe
    if collision_checker.check(vehicle, start_state, grid_map):
        print("Warning: Start invalid, clearing area...")
        generator._clear_rectangular_area(grid_map, start_state, 4.0)

    # 5. Planner
    rrt_planner = RRTPlanner(
        vehicle_model=vehicle,
        collision_checker=collision_checker,
        step_size=3.0,
        max_iterations=50000,
        goal_sample_rate=0.2, 
        goal_threshold=5.0
    )

    debugger = PlanningDebugger()

    # 6. Plan!
    print(f"Planning: {start_state} -> {goal_state}")
    
    import time
    start_time = time.time()
    path = rrt_planner.plan(start_state, goal_state, grid_map, debugger=debugger)
    duration = time.time() - start_time
    
    if not path:
        print("RRT Failed!")
        return

    print(f"RRT Success! Nodes: {len(path)}, Time: {duration:.2f}s")
    
    # 7. Smoothing
    print("Smoothing path...")
    smoother = GreedyShortcutSmoother(vehicle, collision_checker, grid_map)
    
    smooth_start = time.time()
    smoothed_path = smoother.smooth(path, max_iterations=200) # Quick smoothing
    smooth_duration = time.time() - smooth_start
    
    print(f"Smoothing Done! Nodes: {len(smoothed_path)}, Time: {smooth_duration:.2f}s")

    # 8. Visualize
    visualize_result(grid_map, path, smoothed_path, debugger, start_state, goal_state, vehicle)

def visualize_result(grid_map, raw_path, smoothed_path, debugger, start, goal, vehicle):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Background
    ax.imshow(grid_map.data, cmap='Greys', origin='lower', 
              extent=[0, grid_map.width * grid_map.resolution, 
                      0, grid_map.height * grid_map.resolution],
              alpha=0.5)
    
    ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
    ax.plot(goal.x, goal.y, 'rx', markersize=10, label='Goal')
    
    # RRT Tree (Optional, adds clutter but good for debug)
    if hasattr(debugger, 'edges'):
       for edge in debugger.edges:
           s1, s2 = edge
           ax.plot([s1.x, s2.x], [s1.y, s2.y], 'r-', linewidth=0.5, alpha=0.1)

    # Raw Path
    if raw_path:
        px = [s.x for s in raw_path]
        py = [s.y for s in raw_path]
        ax.plot(px, py, 'r--', linewidth=1.5, label='Raw RRT', alpha=0.6)

    # Smoothed Path
    if smoothed_path:
        sx = [s.x for s in smoothed_path]
        sy = [s.y for s in smoothed_path]
        ax.plot(sx, sy, 'b-', linewidth=2.5, label='Smoothed', alpha=0.9)
        
        # Draw vehicle footprint on smoothed path
        sample_interval = max(1, len(smoothed_path) // 15)
        for i in range(0, len(smoothed_path), sample_interval):
            state = smoothed_path[i]
            poly = vehicle.get_visualization_polygon(state)
            patch = Polygon(poly, closed=True, fill=False, edgecolor='blue', alpha=0.5)
            ax.add_patch(patch)
            ax.arrow(state.x, state.y, 1.5 * math.cos(state.theta_rad), 1.5 * math.sin(state.theta_rad),
                     head_width=0.5, head_length=0.8, fc='blue', ec='blue')

    ax.set_title("RRT + Greedy Shortcut Smoothing")
    ax.legend()
    ax.set_aspect('equal')
    
    save_path = os.path.join(os.path.dirname(__file__), "rrt_smoothing_result.png")
    plt.savefig(save_path)
    print(f"Result saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    test_rrt_with_smoothing()
