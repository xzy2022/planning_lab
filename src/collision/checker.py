# src/collision/checker.py
import numpy as np
from typing import List, Tuple

from src.vehicles.base import VehicleBase, State
from src.map.grid_map import GridMap
from .config import CollisionConfig, CollisionMethod
from .geometry import check_sat_polygon_collision, get_grid_aabb_polygon

class CollisionChecker:
    def __init__(self, config: CollisionConfig = None):
        if config is None:
            self.config = CollisionConfig()
        else:
            self.config = config

    def check(self, vehicle: VehicleBase, state: State, grid_map: GridMap) -> bool:
        """
        统一入口：检查特定状态下车辆是否碰撞
        :return: True 表示碰撞 (不安全), False 表示安全
        """
        
        # --- Phase 1: Broad Phase (粗筛) ---
        # 利用 get_bounding_circle (接口 3)
        # 注意：这里的 radius 已经包含了 safe_margin
        bx, by, b_radius = vehicle.get_bounding_circle(state)
        
        # 1.1 地图边界检查
        map_w_m = grid_map.width * grid_map.resolution
        map_h_m = grid_map.height * grid_map.resolution
        if (bx - b_radius < 0 or bx + b_radius > map_w_m or
            by - b_radius < 0 or by + b_radius > map_h_m):
            return True # 出界视为碰撞

        # 1.2 外接圆障碍物粗查
        # 如果外接圆范围内全是空地，则绝对安全，无需进行 Narrow Phase
        if not self._check_circle_in_grid(bx, by, b_radius, grid_map):
            return False 

        # --- Phase 2: Narrow Phase (精筛) ---
        # 如果粗筛发现可能碰撞，则根据配置启用精细检测
        
        if self.config.method == CollisionMethod.CIRCLE_ONLY:
            # 配置为仅用圆检测，那么粗筛结果就是最终结果
            return True 
            
        elif self.config.method == CollisionMethod.MULTI_CIRCLE:
            # 利用 get_collision_circles (接口 1)
            cx_list, cy_list, radius = vehicle.get_collision_circles(state)
            
            # 遍历所有小圆
            for cx, cy in zip(cx_list, cy_list):
                # 只要有一个小圆碰到障碍，就视为碰撞
                if self._check_circle_in_grid(cx, cy, radius, grid_map):
                    return True
            return False
            
        elif self.config.method == CollisionMethod.POLYGON:
            # 利用 get_collision_polygon (接口 2)
            poly_coords = vehicle.get_collision_polygon(state)
            return self._check_polygon_in_grid(poly_coords, grid_map)
            
        return False

    def _check_circle_in_grid(self, cx: float, cy: float, radius: float, grid_map: GridMap) -> bool:
        """
        检查单个圆是否覆盖了任何障碍物网格
        """
        res = grid_map.resolution
        
        # 计算圆的外接矩形对应的网格索引范围
        x_min = int((cx - radius) / res)
        x_max = int((cx + radius) / res)
        y_min = int((cy - radius) / res)
        y_max = int((cy + radius) / res)
        
        r2 = radius**2

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if grid_map.is_obstacle(x, y):
                    # 优化：简单的 AABB 重叠检查 (保守策略)
                    # 如果要精确：检查网格中心或最近点到圆心的距离
                    # 这里采用“网格中心”法
                    cell_cx = (x + 0.5) * res
                    cell_cy = (y + 0.5) * res
                    if (cell_cx - cx)**2 + (cell_cy - cy)**2 <= r2:
                        return True
        return False

    def _check_polygon_in_grid(self, poly_coords: np.ndarray, grid_map: GridMap) -> bool:
        """
        检查多边形是否碰到障碍物
        """
        # 1. 计算多边形 AABB，减少遍历范围
        min_x, min_y = np.min(poly_coords, axis=0)
        max_x, max_y = np.max(poly_coords, axis=0)
        
        x_min_idx = int(min_x / grid_map.resolution)
        x_max_idx = int(max_x / grid_map.resolution)
        y_min_idx = int(min_y / grid_map.resolution)
        y_max_idx = int(max_y / grid_map.resolution)
        
        # 2. 遍历范围内所有障碍物
        for y in range(y_min_idx, y_max_idx + 1):
            for x in range(x_min_idx, x_max_idx + 1):
                if grid_map.is_obstacle(x, y):
                    # 3. SAT 碰撞检测
                    # 将该障碍物网格视为一个矩形多边形
                    grid_poly = get_grid_aabb_polygon(x, y, grid_map.resolution)
                    
                    if check_sat_polygon_collision(poly_coords, grid_poly):
                        return True
        return False