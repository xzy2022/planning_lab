# src/vehicles/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List
from .config import VehicleConfig
import math
import numpy as np
from src.types import State



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

    @abstractmethod
    def propagate_towards(self, start: State, target: State, max_dist: float) -> Tuple[State, List[State]]:
        """
        [新增接口] 尝试从 start 向 target 移动。
        
        Args:
            start: 起点状态
            target: 采样点/目标点状态
            max_dist: 允许生长的最大距离 (RRT step_size)
            
        Returns:
            (final_state, trajectory):
            - final_state: 实际到达的最终状态
            - trajectory: 过程中的轨迹点列表（用于碰撞检测）
        """
        pass

    @abstractmethod
    def get_bounding_circle(self, state: State) -> Tuple[float, float, float]:
        """
        [粗检测接口] 获取能够完全包围车辆的最小外接圆 (Bounding Circle)。
        
        Args:
            state: 车辆当前状态 (x, y, theta)
            
        Returns:
            (center_x, center_y, radius): 
            - center_x, center_y: 外接圆圆心的世界坐标 (注意：通常不等于车辆原点/后轴中心)
            - radius: 外接圆半径 (已包含安全余量)
            
        用途:
            1. MapGenerator: 清除车辆所在位置的障碍物。
            2. CollisionChecker: Broad-phase (粗筛) 碰撞检测，快速剔除绝对安全的车辆。
            3. QuadTree/SpatialHash: 空间索引查询。
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
        c = math.cos(state.theta_rad)
        s = math.sin(state.theta_rad)
        
        # 预分配内存，比临时变量拷贝更快
        world_points = np.empty_like(local_points)
        
        # 向量化计算: x' = x*c - y*s + tx
        world_points[:, 0] = local_points[:, 0] * c - local_points[:, 1] * s + state.x
        world_points[:, 1] = local_points[:, 0] * s + local_points[:, 1] * c + state.y
        
        return world_points