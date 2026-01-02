# src/planning/costs/base.py
from abc import ABC, abstractmethod
from src.types import State

class CostFunction(ABC):
    """
    代价函数基类 (Strategy Interface)
    用于定义从 current 状态移动到 next_node 状态的额外代价。
    """
    @abstractmethod
    def calculate(self, current: State, next_node: State) -> float:
        """
        计算单步移动的额外代价 (不包含基础移动距离代价)
        :param current: 当前状态
        :param next_node: 下一步状态
        :return: 代价数值 (必须 >= 0)
        """
        pass