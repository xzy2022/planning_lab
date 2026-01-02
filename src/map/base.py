# src/map/base.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional

class MapBase(ABC):
    """
    地图抽象基类
    """
    
    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """
        返回地图数据矩阵，通常用于可视化或底层计算。
        约定：0 表示空闲，1 表示障碍物。
        """
        pass
    
    @property
    @abstractmethod
    def resolution(self) -> float:
        """地图分辨率 (m/pixel)"""
        pass
        
    @property
    @abstractmethod
    def width(self) -> int:
        """网格宽度 (x方向数量)"""
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        """网格高度 (y方向数量)"""
        pass

    @abstractmethod
    def is_obstacle(self, x_idx: int, y_idx: int) -> bool:
        """检查特定网格索引是否为障碍物"""
        pass

    @abstractmethod
    def is_obstacle_at_point(self, x: float, y: float) -> bool:
        """检查特定物理位置是否为障碍物"""
        pass

    @abstractmethod
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        [关键接口] 将物理坐标(m)转换为栅格索引
        返回: (x_index, y_index)
        """
        pass

    @abstractmethod
    def grid_to_world(self, x_idx: int, y_idx: int) -> Tuple[float, float]:
        """
        [关键接口] 将栅格索引转换为物理坐标(m)
        通常返回该格子的中心点坐标
        """
        pass
    
    @abstractmethod
    def is_inside(self, x: float, y: float) -> bool:
        """检查物理坐标是否在地图范围内"""
        pass