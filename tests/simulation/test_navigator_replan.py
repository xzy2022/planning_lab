import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.map.grid_map import GridMap
from src.vehicles.ackermann import AckermannVehicle
from src.vehicles.config import AckermannConfig
from src.types import State
from src.collision.checker import CollisionChecker, CollisionConfig, CollisionMethod
from src.planning.planners.hybrid_a_star import HybridAStarPlanner
from src.simulation.navigator import Navigator
from src.simulation.sensor import Sensor

def test_navigator_simulation():
    print("=== Starting Navigator Simulation Test ===")

    # 1. Setup Environment (Global Truth)
    # Reduced size for faster verification
    width, height, res = 60, 60, 1.0
    global_map = GridMap(width=width, height=height, resolution=res)
    
    # Wall from x=25 to 35, y=0 to 40.
    wall_x_min = 25
    wall_x_max = 35
    wall_y_max = 40
    
    for x in range(wall_x_min, wall_x_max):
        for y in range(0, wall_y_max):
             if 0 <= x < width and 0 <= y < height:
                 global_map.data[y, x] = 1

    # 2. Config Vehicle
    vehicle_config = AckermannConfig(
        wheelbase=2.5, 
        max_steer_deg=35.0, 
        width=2.0 
    )
    vehicle = AckermannVehicle(vehicle_config)
    
    # 3. Components
    col_config = CollisionConfig(method=CollisionMethod.POLYGON)
    checker = CollisionChecker(col_config, vehicle, global_map)
    
    planner = HybridAStarPlanner(
        vehicle_model=vehicle,
        collision_checker=checker,
        step_size=2.0, # Larger step for speed
        xy_resolution=1.0, 
        n_steer=3 # Reduced branching
    )
    
    sensor = Sensor(sensing_radius=15.0) 
    
    start = State(5.0, 5.0, 0.0)
    goal = State(55.0, 5.0, 0.0)
    
    navigator = Navigator(global_map, planner, sensor, start, goal, vehicle)
    
    # 4. Run Loop
    print(f"Goal: {goal}")
    print("Starting Navigation...")
    
    success = navigator.navigate(max_steps=300)
    
    # 5. Visualize
    visualize_simulation(global_map, navigator.local_map, navigator.navigated_path, start, goal)
    
    if success:
        print("Simulation Successful!")
    else:
        print("Simulation Failed.")
        sys.exit(1)

def visualize_simulation(global_map, local_map, path, start, goal):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Global Truth
    ax1.imshow(global_map.data, cmap='Greys', origin='lower', alpha=0.6)
    ax1.set_title("Global Map (Truth)")
    
    if path:
        px = [s.x for s in path]
        py = [s.y for s in path]
        ax1.plot(px, py, 'b.-', label='Path')
        
    ax1.plot(start.x, start.y, 'go', markersize=10)
    ax1.plot(goal.x, goal.y, 'rx', markersize=10)
    
    # Local Perception
    # -1 (unknown) might be indistinguishable from 0 (free) if initialized 0.
    # Assuming initialized 0.
    ax2.imshow(local_map.data, cmap='Greys', origin='lower', alpha=0.6)
    ax2.set_title("Local Map (Perception)")
    ax2.plot(px, py, 'b.-')
    
    save_path = os.path.join(os.path.dirname(__file__), "simulation_result.png")
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.close()

if __name__ == "__main__":
    test_navigator_simulation()
