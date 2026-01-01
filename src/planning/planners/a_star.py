# src/planning/planners/a_star.py
import heapq
from typing import List, Tuple, Optional

# 导入之前定义的接口
from src.types import State, Node
from src.planning.planners.base import PlannerBase
from src.vehicles.base import VehicleBase
from src.planning.heuristics.base import Heuristic
from src.planning.costs.base import CostFunction
from src.visualization.debugger import IDebugger, NoOpDebugger
from src.common.collision import CollisionChecker

class AStarPlanner(PlannerBase):
    """
    工业级 A* 规划器实现
    特点：
    1. 策略模式：通过构造函数注入 Heuristic 和 CostFunction，逻辑完全解耦。
    2. 观测模式：通过 plan 方法注入 Debugger，实现 0 开销的条件记录。
    """

    def __init__(self, 
                 vehicle_model: VehicleBase,
                 collision_checker: CollisionChecker, # 碰撞检测器通常也是注入的
                 heuristic: Heuristic,                 # [策略 1] 启发式
                 cost_functions: List[CostFunction],   # [策略 2] 代价函数列表
                 weights: List[float]):                # [策略 3] 对应的权重
        
        self.vehicle = vehicle_model
        self.collision_checker = collision_checker
        self.h_fn = heuristic
        self.cost_fns = cost_functions
        self.weights = weights
        
        # 简单的参数校验
        assert len(self.cost_fns) == len(self.weights), "代价函数数量必须与权重数量一致"

    def plan(self, 
             start: State, 
             goal: State, 
             grid_map,  # Map 具体类型取决于你的 map 模块定义
             debugger: IDebugger = None) -> List[State]:
        
        # --- [切面 1] Debugger 初始化 ---
        # 如果调用者没传 debugger (生产环境)，使用哑巴对象，避免后续大量的 if 判断
        if debugger is None:
            debugger = NoOpDebugger()

        # 记录地图背景数据 (用于可视化底图)
        debugger.set_cost_map(grid_map) 

        # --- [算法核心] 初始化容器 ---
        # 优先级队列，存储 (f_score, state_id, state)
        # Python 的 heapq 是最小堆
        open_set = [] 
        heapq.heappush(open_set, (0.0, start))
        
        # 记录已访问状态及其 "G值" (从起点到当前的实际代价)
        g_scores = {start: 0.0}
        
        # 记录父节点用于回溯路径
        came_from = {start: None}

        # --- [算法核心] 主循环 ---
        while open_set:
            # 1. 弹出 f_score 最小的节点
            current_f, current_state = heapq.heappop(open_set)

            # --- [切面 2] 记录当前扩展节点 ---
            # 这会在图上画出一个“正在探索”的红点
            debugger.record_current_expansion(current_state)

            # 2. 判断是否到达终点 (使用欧氏距离模糊判定)
            if self.h_fn.estimate(current_state, goal) < 1e-2: # 或 vehicle.is_reached(current, goal)
                return self._reconstruct_path(came_from, current_state)

            # 3. 扩展邻居节点 (依赖 Vehicle 模型)
            # get_motion_primitives 返回的是一组可能的下一状态 (x, y, theta)
            neighbors = self.vehicle.get_motion_primitives(current_state)

            for next_state in neighbors:
                # 4. 碰撞检测 (前置剪枝)
                if self.collision_checker.check(self.vehicle, next_state, grid_map):
                    continue

                # 5. [核心逻辑] 计算 Step Cost (G值增量)
                # 遍历所有注入的 cost functions 并加权求和
                step_cost = 0.0
                for fn, w in zip(self.cost_fns, self.weights):
                    step_cost += w * fn.calculate(current_state, next_state)

                # 暂定的新 G 值
                tentative_g = g_scores[current_state] + step_cost

                # 6. 状态更新与入队
                if next_state not in g_scores or tentative_g < g_scores[next_state]:
                    g_scores[next_state] = tentative_g
                    came_from[next_state] = current_state
                    
                    # [核心逻辑] 计算 H 值 (依赖注入的 Heuristic)
                    h_val = self.h_fn.estimate(next_state, goal)
                    f_val = tentative_g + h_val
                    
                    heapq.heappush(open_set, (f_val, next_state))

                    # --- [切面 3] 记录 OpenSet 变化 ---
                    # 这会在图上把此节点标记为绿色 (待探索)
                    debugger.record_open_set_node(next_state, f_val, h_val)

        return [] # 未找到路径

    def _reconstruct_path(self, came_from, current):
        """标准的回溯路径函数"""
        path = [current]
        while current in came_from and came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        return path[::-1]