# src/planning/costs/distance_cost.py
import math
from src.types import State
from .base import CostFunction

class DistanceCost(CostFunction):
    """
    累积路径长度代价。
    Cost = distance * weight
    """
    def calculate(self, current: State, next_node: State) -> float:
        dx = next_node.x - current.x
        dy = next_node.y - current.y
        # 对于车辆模型，通常 distance 就是 v * dt，或者直接算欧氏距离
        return math.hypot(dx, dy)