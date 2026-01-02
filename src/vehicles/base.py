# [接口] 只有抽象基类 (Interface/ABC)
# src/vehicles/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List
from .config import VehicleConfig
import math

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
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """工具函数：基类提供通用数学计算"""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @abstractmethod
    def kinematic_propagate(self, start: State, control: tuple, dt: float) -> State:
        """核心物理推演，留给子类实现"""
        pass