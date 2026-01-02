# src/collision/checker.py
import numpy as np
from typing import List, Tuple

from src.vehicles.base import VehicleBase, State
from src.map.grid_map import GridMap
from .config import CollisionConfig, CollisionMethod
from .geometry import check_sat_polygon_collision, get_grid_aabb_polygon
from .footprint import FootprintModel

class CollisionChecker:
    def __init__(self, config: CollisionConfig = None, vehicle=None, grid_map=None):
        if config is None:
            self.config = CollisionConfig()
        else:
            self.config = config

            # [优化] 如果配置了 RASTER 模式，必须预先初始化 FootprintModel
            if config.method == CollisionMethod.RASTER:
                if vehicle is None or grid_map is None:
                    raise ValueError("Raster mode requires vehicle and map for initialization")
                # 这一步计算量大，只做一次
                self.footprint_model = FootprintModel(vehicle, grid_map.resolution)

    def check(self, vehicle: VehicleBase, state: State, grid_map: GridMap) -> bool:
        """
        统一入口：检查特定状态下车辆是否碰撞
        :return: True 表示碰撞 (不安全), False 表示安全
        """
        
        # --- Phase 1: Broad Phase (粗筛 - 基础几何) ---
        # 利用 get_bounding_circle (接口 3)
        # 注意：这里的 radius 已经包含了 safe_margin
        bx, by, b_radius = vehicle.get_bounding_circle(state)
        
        # 1.1 地图边界检查 (保留：这是 O(1) 的快速检查，且处理越界情况比 Raster 更稳健)
        map_w_m = grid_map.width * grid_map.resolution
        map_h_m = grid_map.height * grid_map.resolution
        if (bx - b_radius < 0 or bx + b_radius > map_w_m or
            by - b_radius < 0 or by + b_radius > map_h_m):
            return True # 出界视为碰撞

        # --- Optimization: Raster Mode Fast Path (针对 Raster 的极速通道) ---
        # [优化] 跳过 Python 循环实现的 _check_circle_in_grid，直接进入 Numpy 查表
        if self.config.method == CollisionMethod.RASTER:
            # 1. 极速查表拿到所有要检查的 (x, y) 索引
            # 注意：需确保 checker 初始化时已构建 self.footprint_model
            indices = self.footprint_model.get_occupied_indices(state)
            
            # 2. 过滤掉越界的索引 (这步可以用 numpy 向量化处理)
            valid_mask = (indices[:, 0] >= 0) & (indices[:, 0] < grid_map.width) & \
                         (indices[:, 1] >= 0) & (indices[:, 1] < grid_map.height)
            valid_indices = indices[valid_mask]
            
            # 如果所有点都在地图外(且没触发1.1)，说明可能在角落缝隙，按安全处理
            # 但通常 1.1 会拦截大部分越界情况
            if len(valid_indices) == 0:
                return False

            # 3. 直接从地图取值 (核心加速点)
            # data[y, x] 注意 numpy 索引顺序通常是 (row, col) 即 (y, x)
            occupied_values = grid_map.data[valid_indices[:, 1], valid_indices[:, 0]]
            
            # 4. 如果有任何一个格子是 1，则碰撞
            if np.any(occupied_values == 1):
                return True
            return False

        # --- Phase 1.2: Broad Phase Obstacle Check (常规粗筛) ---
        # 对于非 Raster 方法，这个 Python 循环比后续的精细检测(如 SAT)要快，所以保留
        if not self._check_circle_in_grid(bx, by, b_radius, grid_map):
            return False 

        # --- Phase 2: Narrow Phase (精细检测 - 其他方法) ---
        
        if self.config.method == CollisionMethod.CIRCLE_ONLY:
            # 配置为仅用圆检测，且通过了1.2的筛选(说明撞了)，则返回 True
            return True 
            
        elif self.config.method == CollisionMethod.MULTI_CIRCLE:
            # 利用 get_collision_circles (接口 1)
            cx_list, cy_list, radius = vehicle.get_collision_circles(state)
            
            # 遍历所有小圆
            for cx, cy in zip(cx_list, cy_list):
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