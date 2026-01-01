# 欧氏距离 (简单 A*)

import math
from .base import Heuristic

class EuclideanHeuristic(Heuristic):
    def estimate(self, current: State, goal: State) -> float:
        return math.hypot(current.x - goal.x, current.y - goal.y)