# src/map/grid_map.py
import numpy as np
import math
from dataclasses import dataclass
from .base import MapBase
from typing import Tuple

class GridMap(MapBase):
    def __init__(self, width: int, height: int, resolution: float = 0.1):
        self._width = width
        self._height = height
        self._resolution = resolution
        self._grid = np.zeros((height, width), dtype=np.int8)  # 初始化全 0 (空闲) 矩阵，类型用 int8 节省内存

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

    def is_obstacle(self, x_idx: int, y_idx: int) -> bool:
        # 边界检查
        if not (0 <= x_idx < self._width and 0 <= y_idx < self._height):
            return True # 越界视为障碍
        return self._grid[y_idx, x_idx] == 1

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

