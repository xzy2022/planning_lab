# src\planning\heuristics\manhattan.py
from src.types import State
from .base import Heuristic

class ManhattanHeuristic(Heuristic):
    """
    曼哈顿距离 (L1).
    Cost = |dx| + |dy|
    注意：在允许对角移动的 Grid Map (8-connected) 中，
    Manhattan (2.0) > Diagonal (1.414)，违反了 Admissibility (h <= true_cost)。
    因此 A* 可能找不到最短路径，但通常能极大加速搜索（贪婪倾向）。
    """
    def estimate(self, current: State, goal: State) -> float:
        return abs(current.x - goal.x) + abs(current.y - goal.y)