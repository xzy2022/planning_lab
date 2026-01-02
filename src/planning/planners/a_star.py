# src/planning/planners/a_star.py
import heapq
import math
from typing import List, Tuple, Dict, Optional

from src.types import State
from src.planning.planners.base import PlannerBase
from src.vehicles.base import VehicleBase
from src.map.grid_map import GridMap
from src.planning.heuristics.base import Heuristic
from src.planning.costs.base import CostFunction
from src.visualization.debugger import IDebugger, NoOpDebugger
from src.collision import CollisionChecker

class AStarPlanner(PlannerBase):
    """
    针对 PointMassVehicle (全向移动/质点模型) 的 Grid A* 实现。
    
    工作流程：
    1. 将连续的 Start/Goal 映射到栅格索引 (Grid Index)。
    2. 使用 8-连通 (8-connected) 方式扩展邻居。
    3. 利用 CollisionChecker 检查离散节点的物理可行性。
    4. 利用 CostFunctions 计算代价值。
    """

    def __init__(self, 
                 vehicle_model: VehicleBase,
                 collision_checker: CollisionChecker,
                 heuristic: Heuristic,
                 cost_functions: List[CostFunction],
                 weights: List[float]):
        
        self.vehicle = vehicle_model
        self.collision_checker = collision_checker
        self.h_fn = heuristic
        self.cost_fns = cost_functions
        self.weights = weights
        
        assert len(self.cost_fns) == len(self.weights), "Cost functions and weights mismatch"

    def plan(self, 
             start: State, 
             goal: State, 
             grid_map: GridMap, 
             debugger: IDebugger = None) -> List[State]:
        
        # 1. 初始化调试器
        if debugger is None:
            debugger = NoOpDebugger()
        debugger.set_cost_map(grid_map)

        # 2. 坐标离散化 (State -> Grid Index)
        # 质点模型在 GridMap 上规划，本质是寻找格子的序列
        start_idx = grid_map.world_to_grid(start.x, start.y)
        goal_idx = grid_map.world_to_grid(goal.x, goal.y)

        # 边界检查：如果起点或终点不在地图内，直接返回
        if not (self._is_valid_index(start_idx, grid_map) and self._is_valid_index(goal_idx, grid_map)):
            print("[A*] Start or Goal is out of map bounds.")
            return []

        # 3. 初始化核心容器
        # OpenSet: 存储 (f_score, x_idx, y_idx)
        # 使用 Tuple 作为 Item，Python 的 heapq 会自动按第一个元素 (f_score) 排序
        open_set = []
        heapq.heappush(open_set, (0.0, start_idx))
        
        # CameFrom: 记录路径回溯链 {child_idx: parent_idx}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {start_idx: None}
        
        # G_Score: 记录从起点到当前点的实际代价 {index: g_val}
        g_scores: Dict[Tuple[int, int], float] = {start_idx: 0.0}

        # 定义 PointMass 的 8 个运动方向 (dx, dy, move_cost_multiplier)
        # 直行代价 1.0，斜行代价 1.414 (sqrt(2))
        motions = [
            (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]

        # 4. 主循环
        while open_set:
            current_f, current_idx = heapq.heappop(open_set)
            
            # --- [Vis] 记录当前扩展 ---
            # 为了可视化和计算 Cost，我们需要把 Index 转回 State
            current_state = self._get_state_from_index(current_idx, grid_map, start.theta_rad)
            debugger.record_current_expansion(current_state)

            # A. 终止条件
            if current_idx == goal_idx:
                return self._reconstruct_path(came_from, current_idx, grid_map, start.theta_rad)

            # B. 扩展邻居
            for dx, dy, cost_mult in motions:
                neighbor_idx = (current_idx[0] + dx, current_idx[1] + dy)

                # B.1 越界检查
                if not self._is_valid_index(neighbor_idx, grid_map):
                    continue

                # B.2 碰撞检测
                # 必须将 Grid Index 转回 物理 State 才能放入 CollisionChecker
                neighbor_state = self._get_state_from_index(neighbor_idx, grid_map, start.theta_rad)
                
                # 如果该位置有碰撞，跳过
                if self.collision_checker.check(self.vehicle, neighbor_state, grid_map):
                    continue

                # B.3 计算 G 值
                # Base Cost: 几何移动距离 (格子数 * 分辨率 * 权重)
                geo_dist = cost_mult * grid_map.resolution
                
                # Extra Cost: 注入的代价函数 (如避障、平滑等)
                # 注意：这里计算的是 "Edge Cost" (从 current 到 neighbor)
                extra_cost = 0.0
                for fn, w in zip(self.cost_fns, self.weights):
                    extra_cost += w * fn.calculate(current_state, neighbor_state)

                new_g = g_scores[current_idx] + geo_dist + extra_cost

                # B.4 更新 OpenSet
                if neighbor_idx not in g_scores or new_g < g_scores[neighbor_idx]:
                    g_scores[neighbor_idx] = new_g
                    
                    # 计算 H 值 (启发式)
                    h_val = self.h_fn.estimate(neighbor_state, goal)
                    f_val = new_g + h_val
                    
                    heapq.heappush(open_set, (f_val, neighbor_idx))
                    came_from[neighbor_idx] = current_idx
                    
                    # --- [Vis] 记录 OpenSet ---
                    debugger.record_open_set_node(neighbor_state, f_val, h_val)

        print("[A*] Open set is empty, no path found.")
        return []

    def _is_valid_index(self, idx: Tuple[int, int], grid_map: GridMap) -> bool:
        """检查索引是否在地图范围内"""
        x, y = idx
        return 0 <= x < grid_map.width and 0 <= y < grid_map.height

    def _get_state_from_index(self, idx: Tuple[int, int], grid_map: GridMap, theta: float) -> State:
        """辅助：将 (ix, iy) 转回 State 对象"""
        wx, wy = grid_map.grid_to_world(idx[0], idx[1])
        # PointMass 不旋转，保持默认 theta 或 0
        return State(wx, wy, theta)

    def _reconstruct_path(self, came_from, current_idx, grid_map, default_theta):
        """从索引回溯并转换为 State 列表"""
        path = []
        while current_idx is not None:
            state = self._get_state_from_index(current_idx, grid_map, default_theta)
            path.append(state)
            current_idx = came_from.get(current_idx)
        return path[::-1]