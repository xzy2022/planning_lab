import math
from src.types import State
from .base import Heuristic

class OctileHeuristic(Heuristic):
    """
    针对 8-连通栅格地图的精确启发式。
    假设直行代价为 1.0，斜行代价为 sqrt(2) ≈ 1.414
    """
    def estimate(self, current: State, goal: State) -> float:
        dx = abs(current.x - goal.x)
        dy = abs(current.y - goal.y)
        # 公式: (sqrt(2) - 1) * min(dx, dy) + max(dx, dy)
        # 0.414 ≈ sqrt(2) - 1
        return 0.41421356 * min(dx, dy) + max(dx, dy)