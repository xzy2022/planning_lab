# [接口] 只有抽象基类 (Interface/ABC)
# src/vehicles/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List

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
    
    @abstractmethod
    def get_shape(self) -> Tuple[float, float]:
        """返回 (width, length)"""
        pass

    @abstractmethod
    def kinematic_propagate(self, start_state: State, control: tuple, dt: float) -> State:
        """输入当前状态和控制量，计算下一时刻状态"""
        pass