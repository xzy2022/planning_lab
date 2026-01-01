# MapBase 接口

# src/map/base.py
from abc import ABC, abstractmethod
import numpy as np

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