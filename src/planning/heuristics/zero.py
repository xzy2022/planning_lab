# src\planning\heuristics\zero.py
from src.types import State
from .base import Heuristic

class ZeroHeuristic(Heuristic):
    """
    零启发式 (h=0).
    这将使 A* 退化为 Dijkstra 算法，保证最优性，但搜索效率最低（向四面八方均匀扩散）。
    """
    def estimate(self, current: State, goal: State) -> float:
        return 0.0