
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.ackermann import AckermannVehicle
from src.vehicles.config import AckermannConfig
from src.types import State
from src.collision.checker import CollisionChecker, CollisionConfig, CollisionMethod

def debug_clearing():
    print("=== Debugging Map Clearing (Benchmark Loops) ===")
    
    seeds = [1010, 1020]
    
    for seed in seeds:
        print(f"\n--- Testing Seed {seed} ---")
        # Setup
        grid_map = GridMap(width=100, height=100, resolution=0.5)
        
        start = State(5.0, 5.0, 0.0)
        goal = State(90.0, 90.0, 0.0)
        
        vehicle_config = AckermannConfig(
            wheelbase=2.5, width=2.0, front_hang=1.0, rear_hang=1.0, safe_margin=0.2
        )
        vehicle = AckermannVehicle(vehicle_config)
        
        plow_config = AckermannConfig(
             wheelbase=2.5, max_steer_deg=35.0, width=2.5, 
             front_hang=1.2, rear_hang=1.2, safe_margin=0.5
        )
        plow_vehicle = AckermannVehicle(plow_config)
    
        generator = MapGenerator(obstacle_density=0.20, inflation_radius_m=0.2, seed=seed)
        generator.generate(grid_map, plow_vehicle, start, goal, extra_paths=6, dead_ends=4)
        
        col_config = CollisionConfig(method=CollisionMethod.POLYGON)
        checker = CollisionChecker(col_config, vehicle, grid_map)
        
        print(f"Collision at start before manual clear: {checker.check(vehicle, start, grid_map)}")
        
        # Manual Clear
        print("Forcing clear attached...")
        generator._clear_rectangular_area(grid_map, start, half_side_m=4.0)
        generator._clear_rectangular_area(grid_map, goal, half_side_m=4.0)
        
        is_colliding = checker.check(vehicle, start, grid_map)
        print(f"Collision at start after manual clear: {is_colliding}")
        
        if is_colliding:
            print("FAILURE: Still colliding.")
            cx = int(start.x / 0.5)
            cy = int(start.y / 0.5)
            print("Surrounding grid (center is start):")
            # Print a larger area to catch edges
            print(grid_map.data[cy-5:cy+6, cx-5:cx+6])
            
            poly = vehicle.get_collision_polygon(start)
            print("Vehicle Polygon points:", poly)
            break # Stop on first failure
        else:
            print("SUCCESS: Area cleared.")

if __name__ == "__main__":
    debug_clearing()
