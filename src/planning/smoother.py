
import random
import copy
from typing import List, Tuple
from src.types import State
from src.vehicles.base import VehicleBase
from src.collision import CollisionChecker
from src.map.grid_map import GridMap

class GreedyShortcutSmoother:
    """
    Greedy Shortcut Smoother for kinematic paths.
    
    It randomly selects two points on the path and attempts to connect them 
    using the vehicle's kinematic propagation. If the new connection is collision-free 
    and kinematically feasible, it replaces the intermediate points.
    """
    def __init__(self, 
                 vehicle: VehicleBase, 
                 collision_checker: CollisionChecker, 
                 grid_map: GridMap):
        self.vehicle = vehicle
        self.collision_checker = collision_checker
        self.grid_map = grid_map

    def smooth(self, path: List[State], max_iterations: int = 100) -> List[State]:
        """
        Smooth the given path.
        
        Args:
            path: Input path (list of States).
            max_iterations: Number of shortcut attempts.
            
        Returns:
            Smoothed path (list of States).
        """
        if not path or len(path) < 3:
            return copy.deepcopy(path)

        optimized_path = copy.deepcopy(path)

        for _ in range(max_iterations):
            # We need at least 3 points to cut a corner, but more practically 
            # we pick two indices i and j.
            if len(optimized_path) <= 2:
                break

            # Pick two random indices
            # i ranges from 0 to len-2
            # j ranges from i+1 to len-1
            # To actually shortcut, we usually want j > i + 1, unless adjacent nodes can be optimized 
            # (re-propagated) which might refine the trajectory but not remove nodes.
            # Let's enforce j > i + 1 to actually remove nodes.
            
            i = random.randint(0, len(optimized_path) - 3)
            j = random.randint(i + 2, len(optimized_path) - 1)
            
            start_state = optimized_path[i]
            target_state = optimized_path[j]
            
            # Try to connect start -> target
            # Note: propagate_towards usually has a step_size/max_dist limit. 
            # If the shortcut is too long, propagate_towards might not reach it fully in one go 
            # if we strictly followed RRT logic, but here we want to see if we can FULLY connect.
            # Ideally, propagate_towards should handle long distance if we give it a large max_dist,
            # OR we loop the propagation until we hit target or collision.
            # 
            # Let's assume propagate_towards tries its best.
            # We pass a large max_dist to allow full connection.
            dist_to_target = ((start_state.x - target_state.x)**2 + (start_state.y - target_state.y)**2)**0.5
            
            # Use a slightly larger max_dist to ensure we cover the gap
            final_state, trajectory = self.vehicle.propagate_towards(
                start=start_state, 
                target=target_state, 
                max_dist=dist_to_target * 1.5
            )
            
            # Check if we actually reached the target (or close enough)
            # AND if the trajectory is collision free
            
            # 1. Reach check
            dist_error = ((final_state.x - target_state.x)**2 + (final_state.y - target_state.y)**2)**0.5
            if dist_error > 0.5: # Tolerance
                continue
                
            # 2. Collision check for the new trajectory
            if self._is_collision_free(trajectory):
                # Success! Replace nodes between i+1 and j-1 with the new trajectory
                # The new path will be: path[:i+1] + new_trajectory_points + path[j+1:]
                # Note: trajectory[0] is start_state (same as path[i]), 
                # and final_state is close to path[j].
                # We should be careful not to duplicate points.
                
                # trajectory includes start (path[i]) so we slice path[:i]
                # trajectory includes end (path[j]) roughly, we can keep path[j] from original to exactness
                # or trust the trajectory end.
                
                # Let's trust trajectory but maybe snap the last point to exact target if needed?
                # Actually, simply splicing:
                # new_segment = trajectory[1:-1] # Exclude start, exclude end (if we want to keep original nodes exact)
                # But kinematic propagation might drift slightly from 'target_state' orientation if not perfect.
                # If we replace path[j] with final_state, subsequent segments (j to j+1) might differ in continuity.
                # So we must ensure continuity.
                # If we modify path[j], we break continuity to path[j+1].
                # Therefore, we can only safely cut if final_state is EXACTLY target_state (unlikely for Ackermann)
                # OR if we re-plan from j onwards (too expensive).
                
                # Correction for Kinematic Smoothing:
                # We can't just snap to an arbitrary state in the middle of a chain unless it matches perfectly.
                # HOWEVER, simpler approach:
                # We are replacing the PATH. So the new node sequence defines the path.
                # The segment path[i] -> path[j] is replaced by the generated trajectory.
                # The next segment would be from the NEW path[new_j] (which was old path[j]) to path[j+1].
                # But the NEW path[new_j] comes from simulation, so it has a specific heading.
                # The original path[j] had a specific heading.
                # If they differ, the segment path[j]->path[j+1] is invalid!
                
                # This is a classic problem with Kinematic Smoothing.
                # "Rewiring" is hard. 
                # BUT, if we are just shortening, maybe we don't care about "exact" node preservation?
                # Actually, if we change the state at j (to final_state), we technically invalidate the edge j->j+1.
                # 
                # OPTION 1: Only smooth if we can match the target state (x,y,theta) very closely.
                # This is hard for simple propagate_towards.
                # 
                # OPTION 2: Re-simulate from j onwards? (Cascading changes - expensive).
                # 
                # OPTION 3: Just accept the discontinuity? No, that's bad for tracking.
                # 
                # OPTION 4: (Common Engineering Hack)
                # Only perform smoothing that lands us "close enough" to the target state 
                # AND we accept that the path might be slightly discontinuous in heading 
                # OR we only do it if the heading match is tight.
                
                # Let's check Angle Difference too.
                angle_diff = abs(final_state.theta_rad - target_state.theta_rad)
                angle_diff = (angle_diff + 3.14159) % (2 * 3.14159) - 3.14159
                
                if abs(dist_error) < 0.5 and abs(angle_diff) < 0.2: # Allow small errors
                    # Optimization Check: Only swap if the new trajectory is SHORTER
                    # Calculate old length
                    # Segment is from optimized_path[i] to optimized_path[j+1]? 
                    # We are replacing nodes from i+1 to j (inclusive of j implicitly if we use trajectory end)
                    # The nodes involved in old path are optimized_path[i...j].
                    # Wait, we connect path[i] to path[j].
                    # So we compare length(path[i]...path[j]) vs length(trajectory)
                    
                    old_segment = optimized_path[i : j+1]
                    old_len = self._calculate_path_length(old_segment)
                    new_len = self._calculate_path_length(trajectory)
                    
                    if new_len < old_len:
                         # Splice!
                        new_segment = trajectory[1:]
                        optimized_path[i+1 : j+1] = new_segment
            else:
                pass # Collision or failure
                
        return optimized_path

    def _calculate_path_length(self, path: List[State]) -> float:
        length = 0.0
        for k in range(len(path) - 1):
            dx = path[k+1].x - path[k].x
            dy = path[k+1].y - path[k].y
            length += (dx*dx + dy*dy)**0.5
        return length

    def _is_collision_free(self, trajectory: List[State]) -> bool:
        for state in trajectory:
            if self.collision_checker.check(self.vehicle, state, self.grid_map):
                return False
        return True
