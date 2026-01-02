# [接口] 只有抽象基类 (Interface/ABC)
# src/vehicles/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List
from .config import VehicleConfig
import math
import numpy as np

# 定义通用的数据结构，方便作为接口的输入输出
@dataclass
class State:
    x: float
    y: float
    theta: float

class VehicleBase(ABC):
    """
    车辆接口基类
    """
    def __init__(self, config: VehicleConfig):
        self.config = config


    @abstractmethod
    def kinematic_propagate(self, start: State, control: tuple, dt: float) -> State:
        """核心物理推演，留给子类实现"""
        pass

    @property
    @abstractmethod
    def clearance_radius(self) -> float:
        """
        [粗检测接口]
        返回一个能完全包围车辆的外接圆半径 (bounding circle radius)。
        用途：MapGenerator 清除障碍、四叉树粗略剔除、KDTree 查询等。
        """
        pass

    def get_collision_circles(self, state: State) -> Tuple[np.ndarray, np.ndarray, float]:
            """
            [精检测接口 - 多圆]
            可选实现。如果子类支持多圆检测，返回 (cx_list, cy_list, radius)。
            """
            raise NotImplementedError("此车辆模型未配置多圆碰撞几何体")

    def get_collision_polygon(self, state: State) -> np.ndarray:
        """
        [精检测接口 - 多边形]
        返回世界坐标系下的多边形顶点 (N, 2)。
        用途：SAT 碰撞检测、可视化绘图。
        """
        raise NotImplementedError("此车辆模型未配置多边形碰撞几何体")

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """工具函数：基类提供通用数学计算"""
        return (angle + math.pi) % (2 * math.pi) - math.pi
    
    @staticmethod
    def transform_points(local_points: np.ndarray, state: State) -> np.ndarray:
        """
        [通用工具] 将局部坐标点变换到世界坐标系
        :param local_points: (N, 2) 数组
        :param state: 车辆状态 (x, y, theta)
        """
        c = math.cos(state.theta)
        s = math.sin(state.theta)
        
        # 预分配内存，比临时变量拷贝更快
        world_points = np.empty_like(local_points)
        
        # 向量化计算: x' = x*c - y*s + tx
        world_points[:, 0] = local_points[:, 0] * c - local_points[:, 1] * s + state.x
        world_points[:, 1] = local_points[:, 0] * s + local_points[:, 1] * c + state.y
        
        return world_points