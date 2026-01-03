# src/map/grid_map.py
import numpy as np
import math
from dataclasses import dataclass
from .base import MapBase
from typing import Tuple
from scipy.ndimage import distance_transform_edt

class GridMap(MapBase):
    def __init__(self, width: int, height: int, resolution: float = 0.1):
        self._width = width
        self._height = height
        self._resolution = resolution
        self._grid = np.zeros((height, width), dtype=np.int8)  # 初始化全 0 (空闲) 矩阵，类型用 int8 节省内存
        self._dist_map = None 

    @property
    def data(self) -> np.ndarray:
        return self._grid

    @property
    def resolution(self) -> float:
        return self._resolution

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height


    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
            """
            物理坐标 -> 栅格索引
            向下取整：floor(x / res)
            """
            x_idx = int(x / self._resolution)
            y_idx = int(y / self._resolution)
            return x_idx, y_idx

    def grid_to_world(self, x_idx: int, y_idx: int) -> Tuple[float, float]:
        """
        栅格索引 -> 物理坐标
        返回格子中心：idx * res + res/2
        """
        x = x_idx * self._resolution + self._resolution / 2.0
        y = y_idx * self._resolution + self._resolution / 2.0
        return x, y

    def is_inside(self, x: float, y: float) -> bool:
        """判断物理坐标是否在地图范围内"""
        xi, yi = self.world_to_grid(x, y)
        return self._is_valid_index(xi, yi)

    def _is_valid_index(self, x_idx: int, y_idx: int) -> bool:
        """内部辅助：检查索引边界"""
        return (0 <= x_idx < self._width) and (0 <= y_idx < self._height)

    def is_obstacle(self, x_idx: int, y_idx: int) -> bool:
        """查询栅格索引是否为障碍"""
        if not self._is_valid_index(x_idx, y_idx):
            return True # 越界通常视为障碍
        return self._grid[y_idx, x_idx] == 1

    def is_obstacle_at_point(self, x: float, y: float) -> bool:
        """直接查询物理点是否是障碍"""
        ix, iy = self.world_to_grid(x, y)
        return self.is_obstacle(ix, iy)

    def precompute_distance_map(self):
        """
        计算欧氏距离变换 (Euclidean Distance Transform, EDT)。
        结果存储在 self._dist_map 中，单位为米。
        """
        # 1. 构造二值反转矩阵
        # distance_transform_edt 计算的是“当前像素离最近的0值像素的距离”
        # 所以我们需要：障碍物=0, 空闲=1
        binary_grid = np.ones_like(self._grid, dtype=float)
        binary_grid[self._grid == 1] = 0  # 障碍物设为 0

        # 2. 计算像素距离
        # output is distance in "number of cells"
        dist_in_cells = distance_transform_edt(binary_grid)

        # 3. 转换为物理距离 (米)
        self._dist_map = dist_in_cells * self._resolution
        
        print(f"[GridMap] Distance map pre-computed. Max clearance: {np.max(self._dist_map):.2f}m")

    def get_obstacle_distance(self, x: float, y: float) -> float:
        """
        获取指定坐标离最近障碍物的距离 (米)。
        如果未预计算或越界，返回 0.0 (视为贴着障碍物/最危险)。
        """
        if self._dist_map is None:
            # 懒加载保护，或者直接报错提示用户先调用 precompute
            print("Warning: Distance map not computed! Call precompute_distance_map() first.")
            return 0.0

        ix, iy = self.world_to_grid(x, y)
        
        if not self._is_valid_index(ix, iy):
            return 0.0 # 越界通常很危险
        
        return self._dist_map[iy, ix]