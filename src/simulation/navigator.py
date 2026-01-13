import copy
from typing import List, Optional, Tuple

from src.types import State
from src.map.grid_map import GridMap
from src.vehicles.ackermann import AckermannVehicle
from src.planning.planners.base import PlannerBase
from src.simulation.sensor import Sensor

class Navigator:
    """
    Simulates an autonomous agent navigating partially observable environment.
    """
    def __init__(self, 
                 global_map: GridMap, 
                 planner: PlannerBase, 
                 sensor: Sensor,
                 start: State,
                 goal: State,
                 vehicle: AckermannVehicle):
        
        self.global_map = global_map
        self.planner = planner
        self.sensor = sensor
        self.start = start
        self.goal = goal
        self.vehicle = vehicle
        
        # Local map initialized as empty (or all free/unknown)
        # We assume 0 is free. Unknown logic depends on planner.
        # Here we initialize as empty GridMap of same size
        self.local_map = GridMap(
            width=global_map.width, 
            height=global_map.height, 
            resolution=global_map.resolution
        )
        # However, purely empty might mean "all free".
        
        self.current_state = start
        self.path: List[State] = []
        self.navigated_path: List[State] = [start] # History
        
        # Statistics
        self.replan_count = 0
        self.step_count = 0
    
    def navigate(self, max_steps: int = 1000) -> bool:
        """
        Main execution loop.
        """
        # Initial Sense
        self.sensor.scan(self.global_map, self.local_map, self.current_state)
        
        # Initial Plan
        if not self._replan():
            print("Initial plan failed!")
            return False
        
        for i in range(max_steps):
            if i % 10 == 0:
                print(f"Step {i}/{max_steps} | Replan: {self.replan_count} | Pos: ({self.current_state.x:.1f}, {self.current_state.y:.1f})")
            self.step_count += 1
            
            # Check Goal Reach
            if self._is_goal_reached():
                print(f"Goal Reached in {self.step_count} steps!")
                return True
            
            # 1. Sense (Simulate perception update as we move)
            # Actually we usually Move then Sense. But let's say continuous.
            # update map at current location
            self.sensor.scan(self.global_map, self.local_map, self.current_state)
            
            # 2. Check Path Validity
            if self._check_collision_on_path():
                print(f"Path blocked at step {i}! Replanning...")
                if not self._replan():
                    print("Replanning failed! Stuck.")
                    return False
            
            # 3. Act (Move one step along path)
            if not self.path:
                print("No path to follow!")
                return False
            
            # Pop next state
            next_state = self.path.pop(0)
            
            # Execute move (teleport for now, or kinematic propagate if we want detailed control)
            # Here we trust the planner's state
            self.current_state = next_state
            self.navigated_path.append(next_state)
            
        print("Max steps reached.")
        return False

    def _replan(self) -> bool:
        """
        Triggers planner on local map.
        """
        self.replan_count += 1
        print(f"Replanning from {self.current_state}...")
        new_path = self.planner.plan(self.current_state, self.goal, self.local_map)
        if new_path:
            self.path = new_path
            return True
        return False
        
    def _check_collision_on_path(self) -> bool:
        """
        Checks if the remaining path collides with *known* obstacles in local_map.
        """
        # We assume the planner provided states
        # The vehicle/collision_checker needs to be accessible.
        # Ideally planner has collision_checker, or we hold one.
        # Let's fallback to planner's collision checker if public, or create one.
        
        # Accessing private member of planner is risky.
        # But HybridAStarPlanner has self.collision_checker
        checker = getattr(self.planner, 'collision_checker', None)
        if not checker:
             raise ValueError("Planner must have a collision_checker attribute")
             
        # Check immediate next few steps or full path?
        # Checking full path is safer.
        for state in self.path:
             if checker.check(self.vehicle, state, self.local_map):
                 return True
        return False

    def _is_goal_reached(self) -> bool:
        # Distance check
        dist = (self.current_state.x - self.goal.x)**2 + (self.current_state.y - self.goal.y)**2
        return dist < 2.0**2 # 2m tolerance
