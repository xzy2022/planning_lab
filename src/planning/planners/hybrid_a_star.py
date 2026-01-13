import heapq
import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Set

from src.types import State
from src.planning.planners.base import PlannerBase
from src.vehicles.ackermann import AckermannVehicle
from src.map.grid_map import GridMap
from src.visualization.debugger import IDebugger, NoOpDebugger
from src.collision.checker import CollisionChecker

class HybridNode:
    def __init__(self, 
                 grid_idx: Tuple[int, int, int], 
                 state: State, 
                 g_cost: float, 
                 f_cost: float, 
                 parent_idx: Optional[Tuple[int, int, int]] = None,
                 direction: int = 1):
        self.grid_idx = grid_idx  # (ix, iy, itheta)
        self.state = state        # Continuous state (x, y, theta)
        self.g_cost = g_cost
        self.f_cost = f_cost
        self.parent_idx = parent_idx
        self.direction = direction # 1 for forward, -1 for reverse

    def __lt__(self, other):
        return self.f_cost < other.f_cost

class HybridAStarPlanner(PlannerBase):
    """
    Hybrid A* Planner for Ackermann Vehicles.
    Combines discrete heuristics with continuous kinematic expansion.
    """

    def __init__(self, 
                 vehicle_model: AckermannVehicle,
                 collision_checker: CollisionChecker,
                 xy_resolution: float = 0.5,
                 theta_resolution: float = np.deg2rad(5.0),
                 step_size: float = 0.5, # Integration step / arc length
                 n_steer: int = 5, # Number of steering angles to sample
                 lambda_reverse: float = 2.0, # Penalty multiplier for reverse driving
                 lambda_steer: float = 1.0, # Penalty multiplier for steering usage
                 lambda_switch: float = 5.0, # Penalty for switching direction
                 analytic_expansion_ratio: float = 0.2): # Prob. of trying analytic expansion
        
        self.vehicle = vehicle_model
        self.collision_checker = collision_checker
        self.xy_resol = xy_resolution
        self.theta_resol = theta_resolution
        self.step_size = step_size
        self.n_steer = n_steer # e.g. 5: -max, -max/2, 0, max/2, max
        
        # Costs
        self.lambda_reverse = lambda_reverse
        self.lambda_steer = lambda_steer
        self.lambda_switch = lambda_switch
        
        self.analytic_ratio = analytic_expansion_ratio

    def plan(self, 
             start: State, 
             goal: State, 
             grid_map: GridMap, 
             debugger: IDebugger = None) -> List[State]:
        print(f"[HybridA*] Plan requested: {start} -> {goal}")
        
        if debugger is None:
            debugger = NoOpDebugger()
        debugger.set_cost_map(grid_map)

        # 1. Initialization
        start_idx = self._get_index(start)
        goal_idx = self._get_index(goal)
        
        # Check start/goal validity
        if self.collision_checker.check(self.vehicle, start, grid_map):
            print("[HybridA*] Start state is in collision!")
            return []
        if self.collision_checker.check(self.vehicle, goal, grid_map):
            print("[HybridA*] Goal state is in collision!")
            return []

        # OpenSet: Priority Queue of HybridNode
        open_set = []
        start_node = HybridNode(start_idx, start, 0.0, self._calc_heuristic(start, goal), None)
        heapq.heappush(open_set, start_node)
        
        # ClosedSet / LookUp: Map grid_idx -> HybridNode (to track best g_cost)
        closed_set: Dict[Tuple[int, int, int], HybridNode] = {start_idx: start_node}
        
        iterations = 0
        max_iterations = 50000

        while open_set:
            iterations += 1
            if iterations > max_iterations:
                print("[HybridA*] Max iterations reached.")
                break

            current_node = heapq.heappop(open_set)
            
            # Record expansion
            debugger.record_current_expansion(current_node.state)

            # Lazy valid check (if we found a better path to this node already in closed set, skip)
            # Actually with A*, if we pop it, it's the best path so far. 
            # But we might need to re-expand if we found a better g to an existing node?
            # Standard A*: if current_node.g > closed_set[current_node.grid_idx].g: continue
            # However, in Hybrid A*, grid_idx is a discretization bucket. 
            # We treat the bucket as 'visited' once popped.
            # Ideally we only keep one node per bucket that is "best" representation?
            # Simply: If we already have a closed node for this index with LOWER g, skip.
            # Note: The 'start_node' was put in closed_set. If we popped it, it matches.
            # If we approach same cell with worse cost, ignore.
            if current_node.grid_idx in closed_set and closed_set[current_node.grid_idx].g_cost < current_node.g_cost - 1e-5:
                continue

            # 2. Analytic Expansion (Try to connect to goal directly)
            # Simple Euclidean check or just random probability
            dist_to_goal = math.hypot(current_node.state.x - goal.x, current_node.state.y - goal.y)
            
            # If close enough, try Reeds-Shepp or simple propagate
            if dist_to_goal < 15.0: # Magic number for "close enough" to try analytic
                 # Currently reusing propagate_towards from vehicle which might be simple steering
                 # Ideally this should be Reeds-Shepp. For now, let's trust vehicle.propagate_towards
                 # But vehicle.propagate_towards is a local controller simulation, which is fine.
                 final_state, trajectory = self.vehicle.propagate_towards(current_node.state, goal, max_dist=dist_to_goal + 2.0)
                 
                 # Check if reached goal tolerances
                 if self._is_goal_reached(final_state, goal):
                     # Check collision for the trajectory
                     if self._check_path_collision(trajectory, grid_map):
                         # Found a path!
                         # Construct full path and return
                         return self._reconstruct_path(current_node, trajectory, closed_set)

            # 3. Kinematic Expansion
            next_nodes = self._get_next_nodes(current_node, goal)
            
            for next_node in next_nodes:
                # Boundary Check
                if not grid_map.is_inside(next_node.state.x, next_node.state.y):
                    continue
                    
                # Collision Check
                if self.collision_checker.check(self.vehicle, next_node.state, grid_map):
                    continue
                
                # Check if this node is better
                idx = next_node.grid_idx
                if idx not in closed_set or next_node.g_cost < closed_set[idx].g_cost:
                    closed_set[idx] = next_node
                    heapq.heappush(open_set, next_node)
                    
                    # Debug
                    debugger.record_open_set_node(next_node.state, next_node.f_cost, next_node.f_cost - next_node.g_cost)

        return []

    def _get_next_nodes(self, current_node: HybridNode, goal: State) -> List[HybridNode]:
        nodes = []
        
        # Steering Sampling
        # e.g. [-max, -max*0.5, 0, max*0.5, max]
        max_steer = self.vehicle.config.max_steer
        steers = np.linspace(-max_steer, max_steer, self.n_steer)
        # print(f"Steer: {steers}")
        
        # Directions [1, -1]
        directions = [1, -1]
        
        dt = 0.1 # Integration step time? Or define step_size as distance.
        # If propagate takes (v, steer, time), we need to ensure arc length is approx step_size.
        # Arc L = v * t. If v=1, t=step_size.
        # Re-using vehicle kinematic model which takes (v, steering).
        
        # Speed assumption: 1.0 m/s for normalized calculation
        v_mag = 1.0 
        
        # Integration time to achieve step_size euclidean approx
        # For small steps, arc length ~= chord length.
        integration_time = self.step_size / v_mag

        for d in directions:
            for delta in steers:
                v = v_mag * d
                
                # Kinematic Propagate
                next_state = self.vehicle.kinematic_propagate(current_node.state, (v, delta), integration_time)
                
                # Calculate Costs
                # Base step cost
                step_cost = self.step_size
                
                # Reverse penalty
                if d < 0:
                    step_cost *= self.lambda_reverse
                
                # Steering penalty
                step_cost += abs(delta) * self.lambda_steer
                
                # Switch direction penalty
                if d != current_node.direction:
                     step_cost += self.lambda_switch
                
                new_g = current_node.g_cost + step_cost
                new_h = self._calc_heuristic(next_state, goal)
                
                next_node = HybridNode(
                    self._get_index(next_state),
                    next_state,
                    new_g,
                    new_g + new_h,
                    current_node.grid_idx,
                    d
                )
                nodes.append(next_node)
                
        return nodes

    def _get_index(self, state: State) -> Tuple[int, int, int]:
        ix = int(round(state.x / self.xy_resol))
        iy = int(round(state.y / self.xy_resol))
        itheta = int(round(self.vehicle.normalize_angle(state.theta_rad) / self.theta_resol))
        return (ix, iy, itheta)

    def _calc_heuristic(self, state: State, goal: State) -> float:
        # Simple Euclidean for now. 
        # Ideally: Non-holonomic-without-obstacles (Reeds-Shepp) max(Euclidean, RS)
        # But let's start simple.
        return math.hypot(state.x - goal.x, state.y - goal.y)

    def _is_goal_reached(self, state: State, goal: State) -> bool:
        # Check if we are close enough to goal
        dist = math.hypot(state.x - goal.x, state.y - goal.y)
        angle_diff = abs(self.vehicle.normalize_angle(state.theta_rad - goal.theta_rad))
        return dist < 1.0 and angle_diff < np.deg2rad(15)

    def _check_path_collision(self, path: List[State], grid_map: GridMap) -> bool:
        for s in path:
            if not grid_map.is_inside(s.x, s.y):
                return False # Actually if out of bounds it's invalid
            if self.collision_checker.check(self.vehicle, s, grid_map):
                return False
        return True

    def _reconstruct_path(self, current_node: HybridNode, analytic_path: List[State], closed_set: Dict) -> List[State]:
        # analytic_path is from current_node.state to goal
        # We need to trace back from current_node to start
        
        path_states = []
        
        # Traceback
        curr = current_node
        while curr is not None:
            path_states.append(curr.state)
            if curr.parent_idx is None:
                break
            curr = closed_set.get(curr.parent_idx)
            
        path_states = path_states[::-1] # Reverse to Start -> Current
        
        # Append analytic path (excluding duplicate start point if needed)
        # analytic_path[0] is current_node.state
        if analytic_path:
             path_states.extend(analytic_path[1:])
             
        return path_states
