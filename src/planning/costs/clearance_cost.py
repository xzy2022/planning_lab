# src\planning\costs\clearance_cost.py
from src.types import State
from src.planning.costs.base import CostFunction
from src.map.grid_map import GridMap
import math

class ClearanceCost(CostFunction):
    """
    风险代价：利用预计算的 Distance Map 让车辆远离障碍物。
    """
    def __init__(self, grid_map: GridMap, risk_dist: float = 2.0, weight_factor: float = 10.0):
        """
        :param risk_dist: 超过这个距离就认为安全了，Cost为0 (米)
        :param weight_factor: 代价的缩放系数
        """
        self.grid_map = grid_map
        self.risk_dist = risk_dist
        self.weight_factor = weight_factor
        
        # 确保地图已经计算过距离场
        if self.grid_map._dist_map is None:
             self.grid_map.precompute_distance_map()

    def calculate(self, current: State, next_node: State) -> float:
        # 1. 查表获取离最近障碍物的距离
        dist = self.grid_map.get_obstacle_distance(next_node.x, next_node.y)
        
        # 2. 如果距离大于安全阈值，无代价
        if dist >= self.risk_dist:
            return 0.0
        
        # 3. 距离越近，代价越高
        # 常用公式: Cost = weight * (safe_dist - actual_dist)
        # 或者指数型: Cost = weight * exp(-actual_dist)
        return self.weight_factor * (self.risk_dist - dist)