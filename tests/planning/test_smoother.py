
import unittest
import sys
import os
import math
import matplotlib.pyplot as plt
from typing import List

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.types import State
from src.map.grid_map import GridMap
from src.vehicles.ackermann import AckermannVehicle
from src.vehicles.config import AckermannConfig
from src.collision.checker import CollisionChecker, CollisionConfig, CollisionMethod
from src.planning.smoother import GreedyShortcutSmoother

class TestGreedySmoother(unittest.TestCase):
    def setUp(self):
        # 1. Setup Map (Open space)
        self.width, self.height, self.res = 100, 100, 0.5
        self.grid_map = GridMap(width=self.width, height=self.height, resolution=self.res)
        
        # 2. Setup Vehicle
        self.vehicle_config = AckermannConfig(
            wheelbase=2.5, max_steer_deg=35.0, width=2.0, 
            front_hang=0.9, rear_hang=0.9, safe_margin=0.2
        )
        self.vehicle = AckermannVehicle(self.vehicle_config)
        
        # 3. Setup Collision Checker
        col_config = CollisionConfig(method=CollisionMethod.POLYGON)
        self.collision_checker = CollisionChecker(col_config, self.vehicle, self.grid_map)
        
        # 4. Smoother
        self.smoother = GreedyShortcutSmoother(self.vehicle, self.collision_checker, self.grid_map)

    def test_smoothing_logic(self):
        print("\n=== Testing Smoothing Logic ===")
        
        # Create a Zig-Zag Path manually
        # A path that goes Up, then Right, then Up
        # Ideally it should just go Diagonal Up-Right
        
        # NOTE: Kinematic constraints! 
        # We need a path that is "feasible" but suboptimal.
        # Let's create a path composed of two propagate_towards calls that form a 'V' shape
        # Start -> Mid -> End
        start = State(10, 10, 0.0) # Facing Right
        mid = State(30, 30, math.pi/2) # Facing Up (Sharp turn required to reach here? Unlikely from 0.0)
        
        # Let's generate a feasible but wiggly path using the vehicle itself
        path = []
        path.append(start)
        
        # Segment 1: Drive straight a bit
        curr = start
        for _ in range(20):
            curr = self.vehicle.kinematic_propagate(curr, (2.0, 0.1), 0.5) # Slight left turn
            path.append(curr)
            
        # Segment 2: Turn right hard
        for _ in range(20):
            curr = self.vehicle.kinematic_propagate(curr, (2.0, -0.6), 0.5) # Hard right
            path.append(curr)
            
        # Segment 3: Turn left hard (Zig Zag back)
        for _ in range(20):
            curr = self.vehicle.kinematic_propagate(curr, (2.0, 0.6), 0.5) # Hard left
            path.append(curr)
            
        original_length = len(path)
        print(f"Original Path Nodes: {original_length}")
        
        # Run Smoother
        # We perform enough iterations to ensure some shortcuts are found
        smoothed_path = self.smoother.smooth(path, max_iterations=500)
        
        smoothed_length = len(smoothed_path)
        print(f"Smoothed Path Nodes: {smoothed_length}")
        
        # Verification
        # 1. Should be shorter (fewer nodes usually implies distance shortcut if resolution is similar, 
        # but here we splice continuous trajectories. 
        # Ideally we compare total arc length, but node count is a decent proxy if dt is constant)
        
        # Let's calculate approx arc length
        def calc_len(p):
            l = 0
            for i in range(len(p)-1):
                l += math.hypot(p[i+1].x - p[i].x, p[i+1].y - p[i].y)
            return l
            
        orig_dist = calc_len(path)
        new_dist = calc_len(smoothed_path)
        print(f"Original Dist: {orig_dist:.2f}m")
        print(f"Smoothed Dist: {new_dist:.2f}m")
        
        self.assertLess(new_dist, orig_dist, "Smoothed path should be shorter")
        
        # Visualization
        self._visualize(path, smoothed_path)

    def _visualize(self, original, smoothed):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.grid_map.data, cmap='Greys', origin='lower', 
                  extent=[0, self.width, 0, self.height], alpha=0.5)
        
        ox = [s.x for s in original]
        oy = [s.y for s in original]
        ax.plot(ox, oy, 'r--', label='Original', alpha=0.5, linewidth=2)
        
        sx = [s.x for s in smoothed]
        sy = [s.y for s in smoothed]
        ax.plot(sx, sy, 'b-', label='Smoothed', linewidth=2)
        
        ax.set_title("Greedy Shortcut Smoother Test")
        ax.legend()
        ax.set_aspect('equal')
        
        save_path = os.path.join(os.path.dirname(__file__), "smoother_result.png")
        plt.savefig(save_path)
        print(f"Result saved to {save_path}")
        plt.close()

if __name__ == "__main__":
    unittest.main()
