# 基础栅格地图

# src/map/grid_map.py
import numpy as np
import math
from dataclasses import dataclass
from .base import MapBase

class GridMap(MapBase):
    def __init__(self, width: int, height: int, resolution: float = 0.1):
        self._width = width
        self._height = height
        self._resolution = resolution
        # 初始化全 0 (空闲) 矩阵，类型用 int8 节省内存
        self._grid = np.zeros((height, width), dtype=np.int8)

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

    def generate_random_obstacles(self, density: float = 0.1, seed: int = None):
        """
        功能1：随机生成障碍物
        :param density: 障碍物密度 (0~1)
        :param seed: 随机种子，用于复现
        """
        if seed is not None:
            np.random.seed(seed)
            
        # 生成随机矩阵，小于阈值的设为 1 (障碍)
        random_mask = np.random.rand(self._height, self._width) < density
        self._grid[random_mask] = 1
        
        # 强制设置四面围墙，防止跑出地图
        self._grid[0, :] = 1
        self._grid[-1, :] = 1
        self._grid[:, 0] = 1
        self._grid[:, -1] = 1

    def inflate_obstacles(self, radius_grids: int):
        """
        功能2：障碍物膨胀 (简单的形态学膨胀)
        这会让分散的障碍物连成片，形成类似迷宫或洞穴的结构。
        """
        if radius_grids <= 0:
            return

        # 获取所有当前障碍物的坐标
        obs_y, obs_x = np.where(self._grid == 1)
        
        # 创建一个副本用于修改
        inflated_grid = self._grid.copy()
        
        rows, cols = self._grid.shape
        
        # 遍历所有障碍物点，将其周围设为障碍物
        # 注意：对于大地图，使用 scipy.ndimage.binary_dilation 会更快，
        # 这里为了减少依赖使用基础 numpy 实现切片操作
        for y, x in zip(obs_y, obs_x):
            y_min = max(0, y - radius_grids)
            y_max = min(rows, y + radius_grids + 1)
            x_min = max(0, x - radius_grids)
            x_max = min(cols, x + radius_grids + 1)
            
            inflated_grid[y_min:y_max, x_min:x_max] = 1
            
        self._grid = inflated_grid