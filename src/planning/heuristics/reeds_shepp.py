# RS 曲线距离 (Hybrid A*)

from .base import Heuristic
# 假设有一个 RS 曲线库
import reeds_shepp_library 

class ReedsSheppHeuristic(Heuristic):
    def __init__(self, turning_radius: float):
        # [关键] 依赖的信息在这里注入，Planner 根本不需要知道 radius 的存在
        self.radius = turning_radius

    def estimate(self, current: State, goal: State) -> float:
        # 内部调用复杂的 RS 曲线库
        length = reeds_shepp_library.path_length(
            current, goal, self.radius
        )
        return length