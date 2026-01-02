# src/planning/heuristics/euclidean.py
import math
from src.types import State  
from .base import Heuristic

class EuclideanHeuristic(Heuristic):
    """
    欧氏距离启发式 (Holonomic Heuristic)
    适用于：
    1. 质点模型 (PointMass)
    2. 对非完整约束车辆 (Ackermann) 的粗略估计 (Admissible但可能Loose)
    """
    def estimate(self, current: State, goal: State) -> float:
        return math.hypot(current.x - goal.x, current.y - goal.y)