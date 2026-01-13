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
from src.visualization.debugger import PlanningDebugger
from src.planning.planners import HybridAStarPlanner

def test_hybrid_a_star_planning():
    print("=== Starting Hybrid A* Planning Test ===")

    # 1. Initialize Map
    width, height, res = 200, 200, 0.5
    grid_map = GridMap(width=width, height=height, resolution=res)
    
    # 2. Configure Vehicle
    vehicle_config = AckermannConfig(
        wheelbase=2.5, 
        max_steer_deg=35.0, 
        width=2.0, 
        front_hang=0.9, 
        rear_hang=0.9,
        safe_margin=0.2
    )
    vehicle = AckermannVehicle(vehicle_config)
    
    # 3. Generate Map with "Plow" Vehicle
    plow_config = AckermannConfig(
        wheelbase=2.5, 
        max_steer_deg=35.0, 
        width=2.5, 
        front_hang=1.2, 
        rear_hang=1.2,
        safe_margin=0.5
    )
    plow_vehicle = AckermannVehicle(plow_config)

    print("Generating map...")
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

    if collision_checker.check(vehicle, start_state, grid_map):
        print("Warning: Start state collision! Clearing area...")
        generator._clear_rectangular_area(grid_map, start_state, 4.0)

    if collision_checker.check(vehicle, goal_state, grid_map):
        print("Warning: Goal state collision! Clearing area...")
        generator._clear_rectangular_area(grid_map, goal_state, 4.0)

    # 5. Hybrid A* Planner
    planner = HybridAStarPlanner(
        vehicle_model=vehicle,
        collision_checker=collision_checker,
        xy_resolution=1.0, 
        theta_resolution=np.deg2rad(5.0),
        step_size=1.0, # Reduced step size for better maneuverability
        n_steer=5,
        analytic_expansion_ratio=0.1 
        # 解析展开，每一次扩展节点有一定概率直接解析求解变换到终点的路径，并判断是否碰撞
    )

    debugger = PlanningDebugger()

    # 6. Execute Planning
    print(f"Planning: {start_state} -> {goal_state}")
    
    import time
    start_time = time.time()
    
    path = planner.plan(start_state, goal_state, grid_map, debugger=debugger)
    
    duration = time.time() - start_time
    print(f"Planning Time: {duration:.2f}s")
    
    if not path:
        print("Planning Failed!")
    else:
        print(f"Planning Success! Path length: {len(path)}")
        
    # 7. Visualize
    visualize_result(grid_map, path, debugger, start_state, goal_state, vehicle)

def visualize_result(grid_map, path, debugger, start, goal, vehicle):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Background
    ax.imshow(grid_map.data, cmap='Greys', origin='lower', 
              extent=[0, grid_map.width * grid_map.resolution, 
                      0, grid_map.height * grid_map.resolution],
              alpha=0.5)
    
    ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
    ax.plot(goal.x, goal.y, 'rx', markersize=10, label='Goal')
    
    # Expanded Nodes (from debugger)
    # debugger.expanded_nodes is a list of States
    if hasattr(debugger, 'expanded_nodes'):
        ex_x = [s.x for s in debugger.expanded_nodes]
        ex_y = [s.y for s in debugger.expanded_nodes]
        ax.plot(ex_x, ex_y, 'c.', markersize=1, alpha=0.3, label='Expanded')

    # Result Path
    if path:
        path_x = [s.x for s in path]
        path_y = [s.y for s in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Hybrid A* Path')
        
        # Vehicle Footprint
        sample_interval = max(1, len(path) // 20)
        for i in range(0, len(path), sample_interval):
            state = path[i]
            poly = vehicle.get_visualization_polygon(state)
            patch = Polygon(poly, closed=True, fill=False, edgecolor='blue', alpha=0.5)
            ax.add_patch(patch)
            
            # Heading Arrow
            ax.arrow(state.x, state.y, 1.5 * math.cos(state.theta_rad), 1.5 * math.sin(state.theta_rad),
                     head_width=0.5, head_length=0.8, fc='blue', ec='blue')

    ax.set_title("Hybrid A* Planning")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_aspect('equal')
    
    save_path = os.path.join(os.path.dirname(__file__), "hybrid_a_star_result.png")
    plt.savefig(save_path)
    print(f"Result saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    test_hybrid_a_star_planning()
