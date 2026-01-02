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

    # def generate_random_obstacles(self, density: float = 0.1, seed: int = None):
    #     """
    #     功能1：随机生成障碍物
    #     :param density: 障碍物密度 (0~1)
    #     :param seed: 随机种子，用于复现
    #     """
    #     if seed is not None:
    #         np.random.seed(seed)
            
    #     # 生成随机矩阵，小于阈值的设为 1 (障碍)
    #     random_mask = np.random.rand(self._height, self._width) < density
    #     self._grid[random_mask] = 1
        
    #     # 强制设置四面围墙，防止跑出地图
    #     self._grid[0, :] = 1
    #     self._grid[-1, :] = 1
    #     self._grid[:, 0] = 1
    #     self._grid[:, -1] = 1

    # def inflate_obstacles(self, radius_grids: int):
    #     """
    #     功能2：障碍物膨胀 (简单的形态学膨胀)
    #     这会让分散的障碍物连成片，形成类似迷宫或洞穴的结构。
    #     """
    #     if radius_grids <= 0:
    #         return

    #     # 获取所有当前障碍物的坐标
    #     obs_y, obs_x = np.where(self._grid == 1)
        
    #     # 创建一个副本用于修改
    #     inflated_grid = self._grid.copy()
        
    #     rows, cols = self._grid.shape
        
    #     # 遍历所有障碍物点，将其周围设为障碍物
    #     # 注意：对于大地图，使用 scipy.ndimage.binary_dilation 会更快，
    #     # 这里为了减少依赖使用基础 numpy 实现切片操作
    #     for y, x in zip(obs_y, obs_x):
    #         y_min = max(0, y - radius_grids)
    #         y_max = min(rows, y + radius_grids + 1)
    #         x_min = max(0, x - radius_grids)
    #         x_max = min(cols, x + radius_grids + 1)
            
    #         inflated_grid[y_min:y_max, x_min:x_max] = 1
            
    #     self._grid = inflated_grid