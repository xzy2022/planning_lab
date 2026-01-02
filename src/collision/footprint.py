# src/collision/footprint.py
import numpy as np
import math
from src.vehicles.base import VehicleBase, State

class FootprintModel:
    def __init__(self, vehicle: VehicleBase, resolution: float, angle_step_deg: float = 2.0):
        self.resolution = resolution
        self.angle_step = math.radians(angle_step_deg)
        self.num_bins = int(360 / angle_step_deg)
        
        # 核心查找表：Index -> List of (dx, dy)
        # self.lookup_table[angle_idx] = np.array([[dx, dy], ...])
        self.lookup_table = []
        
        print(f"Pre-computing footprint tables ({self.num_bins} orientations)...")
        self._precompute_table(vehicle)
        
    def _precompute_table(self, vehicle):
        # 临时创建一个全 0 状态，位于原点
        base_state = State(x=0.0, y=0.0, theta_rad=0.0)
        
        for i in range(self.num_bins):
            theta = i * self.angle_step
            # 注意：角度归一化到 -pi ~ pi 或 0 ~ 2pi 取决于你的习惯，这里用 0~2pi 构建索引
            base_state.theta_rad = theta
            
            # 1. 获取该角度下的多边形 (世界坐标 = 局部坐标，因为 x=0, y=0)
            poly = vehicle.get_collision_polygon(base_state)
            
            # 2. 栅格化该多边形，找出所有覆盖的 (dx, dy)
            # 这里需要一个高质量的栅格化算法，简单起见用 AABB + PointInPoly
            indices = self._rasterize_polygon(poly)
            self.lookup_table.append(indices)
            
    def _rasterize_polygon(self, poly_coords):
        """将多边形转为 grid index 列表 (dx, dy)"""
        # 计算 AABB
        min_x, min_y = np.min(poly_coords, axis=0)
        max_x, max_y = np.max(poly_coords, axis=0)
        
        min_ix = int(np.floor(min_x / self.resolution))
        max_ix = int(np.ceil(max_x / self.resolution))
        min_iy = int(np.floor(min_y / self.resolution))
        max_iy = int(np.ceil(max_y / self.resolution))
        
        occupied_indices = []
        
        for y in range(min_iy, max_iy + 1):
            for x in range(min_ix, max_ix + 1):
                # 取网格中心点进行测试
                cx = (x + 0.5) * self.resolution
                cy = (y + 0.5) * self.resolution
                
                # 几何判断 (你的 geometry.py 里应该有 point_in_polygon)
                if self._point_in_polygon(cx, cy, poly_coords):
                    occupied_indices.append((x, y))
                    
        return np.array(occupied_indices, dtype=np.int32)

    def _point_in_polygon(self, x, y, poly):
        # 射线法简易实现
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def get_occupied_indices(self, state: State):
        """
        运行时调用：返回当前状态下覆盖的绝对网格坐标列表
        """
        # 1. 计算角度索引
        theta = state.theta_rad % (2 * math.pi)
        if theta < 0: theta += 2 * math.pi
        angle_idx = int(theta / self.angle_step) % self.num_bins
        
        # 2. 查表拿到 (dx, dy)
        offsets = self.lookup_table[angle_idx] # Shape: (N, 2)
        
        # 3. 计算当前中心的网格坐标
        curr_ix = int(state.x / self.resolution)
        curr_iy = int(state.y / self.resolution)
        
        # 4. 广播加法：绝对坐标 = 中心 + 偏移
        # return: (N, 2) array of [abs_x, abs_y]
        return offsets + np.array([curr_ix, curr_iy])