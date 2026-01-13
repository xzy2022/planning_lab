import numpy as np
from src.map.grid_map import GridMap
from src.types import State

class Sensor:
    """
    Simulates a LiDAR/Camera sensor that updates the local map 
    based on the global (ground truth) map.
    """
    def __init__(self, sensing_radius: float = 30.0):
        self.sensing_radius = sensing_radius

    def scan(self, global_map: GridMap, local_map: GridMap, state: State):
        """
        Updates local_map with information from global_map within sensing_radius.
        
        Optimistic assumption:
        - Everything outside sensing radius is unknown (but often treated as free by optimistic planners).
        - We strictly copy values: Occupied (1), Free (0).
        """
        
        # Determine scan data bounds to minimize iteration
        # Convert radius to grid cells
        r_cells = int(np.ceil(self.sensing_radius / global_map.resolution))
        
        # Vehicle grid position
        vx, vy = global_map.world_to_grid(state.x, state.y)
        
        # Bounding box of sensor
        min_x = max(0, vx - r_cells)
        max_x = min(global_map.width, vx + r_cells + 1)
        min_y = max(0, vy - r_cells)
        max_y = min(global_map.height, vy + r_cells + 1)
        
        # Optimization: Create a mask or just iterate
        # Since r=30m and res=0.5m -> r=60 cells. 120x120 area = 14400 points.
        # fast enough for python loop? Maybe numpy slicing is better.
        
        # Numpy slicing approach (square window, then masking circle)
        # Note: GridMap data is (height, width) usually or (width, height)?
        # Check GridMap implementation. Usually data[y, x] or data[x, y].
        # Let's assume standard numpy: data[row, col] -> data[y, x].
        
        # We need to verify GridMap logic. Assuming data[y, x].
        
        # Get window from global
        window = global_map.data[min_y:max_y, min_x:max_x]
        
        # Create coordinate grid relative to vehicle
        Y, X = np.ogrid[min_y:max_y, min_x:max_x]
        dist_sq = (X - vx)**2 + (Y - vy)**2
        mask = dist_sq <= r_cells**2
        
        # Update local map
        # Local map should maintain knowledge. 
        # Only update what we see.
        local_window = local_map.data[min_y:max_y, min_x:max_x]
        
        # Apply update: where mask is true, copy global to local
        np.putmask(local_window, mask, window)
