# src/planning/planners/rrt.py
import math
import random
import numpy as np
from typing import List, Optional

from src.types import State
from src.planning.planners.base import PlannerBase
from src.vehicles.base import VehicleBase
from src.map.grid_map import GridMap
from src.collision import CollisionChecker
from src.visualization.debugger import IDebugger, NoOpDebugger

class RRTNode:
    """RRT 树的节点内部类"""
    def __init__(self, state: State):
        self.state = state                             # 1. 节点本身的状态 (x, y, theta_rad)
        self.parent: Optional[RRTNode] = None          # 2. 父节点指针 (用于回溯路径)
        self.path_from_parent: List[State] = []        # 3. 连接轨迹 (父节点 -> 当前节点的微观轨迹，多个 State 组成)

    @property
    def x(self): return self.state.x
    @property
    def y(self): return self.state.y

class RRTPlanner(PlannerBase):
    """
    通用 RRT 规划器 (几何/动力学兼容版)
    兼容 PointMass (几何/全向) 和 Ackermann (动力学/非完整)
    """
    def __init__(self, 
                 vehicle_model: VehicleBase,
                 collision_checker: CollisionChecker,
                 step_size: float = 2.0,       # 单次生长最大距离 [m]
                 max_iterations: int = 5000,   # 最大采样次数
                 goal_sample_rate: float = 0.1,# 目标偏置概率
                 goal_threshold: float = 1.0   # 到达判定距离
                 ):
        
        self.vehicle = vehicle_model
        self.collision_checker = collision_checker
        
        self.step_size = step_size
        self.max_iter = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.goal_threshold = goal_threshold
        
        self.node_list: List[RRTNode] = []

    def plan(self, 
             start: State, 
             goal: State, 
             grid_map: GridMap, 
             debugger: IDebugger = None) -> List[State]:
        
        if debugger is None:
            debugger = NoOpDebugger()
        debugger.set_cost_map(grid_map) # 虽然 RRT 不用 CostMap，但为了绘图一致性

        # 1. 初始化树
        start_node = RRTNode(start)
        self.node_list = [start_node]
        
        # Best node tracking for fallback
        best_node = None
        min_dist_to_goal = float('inf')

        print(f"[RRT] Start planning... Max Iter: {self.max_iter}, Step: {self.step_size}")

        for i in range(self.max_iter):
            # 2. 采样 (Sample)
            rnd_state = self._get_random_sample(goal, grid_map)

            # 3. 寻找最近邻 (Nearest)
            nearest_node = self._get_nearest_node(self.node_list, rnd_state)

            # 4. 生长 (Steer / Propagate)
            # [核心修改] 将动力学推演委托给 Vehicle
            new_node = self._steer(nearest_node, rnd_state)

            # 5. 碰撞检测 (Collision Check)
            if self._check_collision(new_node, grid_map):
                continue
            
            # 6. 添加到树
            self.node_list.append(new_node)
            
            # [Vis] 可视化调试
            debugger.record_current_expansion(new_node.state)
            # 绘制树枝
            debugger.record_edge(nearest_node.state, new_node.state)

            # 7. 判断是否到达目标 (区域)
            dist_to_goal = math.hypot(new_node.x - goal.x, new_node.y - goal.y)
            
            # Update best node
            if dist_to_goal < min_dist_to_goal:
                min_dist_to_goal = dist_to_goal
                best_node = new_node

            if dist_to_goal <= self.goal_threshold:
                # [Optimization] Analytic Expansion (尝试直接连接终点)
                # 当进入阈值范围时，尝试"打一枪"看能不能直接连上精确的终点
                # 注意：给 max_dist 一个稍微宽裕的值，确保能覆盖剩余距离
                final_state, trajectory = self.vehicle.propagate_towards(
                    start=new_node.state,
                    target=goal,
                    max_dist=dist_to_goal * 1.5 + 1.0 
                )
                
                # 检查1: 是否真的接近了终点 (因为运动学约束，可能拐不过去)
                final_dist = math.hypot(final_state.x - goal.x, final_state.y - goal.y)
                
                # 检查2: 这段“补枪”路径是否碰撞
                is_safe = not self._check_collision_trajectory(trajectory, grid_map)
                
                if final_dist < 2.0 and is_safe: # 放宽到 2.0m (视觉上已经很近了，比起 5m)
                     print(f"[RRT] Analytic Expansion succeeded! Goal reached.")
                     goal_node = RRTNode(final_state)
                     goal_node.parent = new_node
                     goal_node.path_from_parent = trajectory
                     return self._reconstruct_path(goal_node)
                else:
                    # Log failure to file for debug
                    # import os
                    # os.system(f'python logs/append_log.py "⚠️ [DEBUG]" "RRT" "Analytic Exp Failed: Dist={final_dist:.2f}, Safe={is_safe}"')
                    pass
        
        # Fallback: If we couldn't connect exactly, but found something within threshold, return it.
        # This addresses "RRT Failed" when we were actually close enough.
        if best_node and min_dist_to_goal <= self.goal_threshold:
            print(f"[RRT] Exact connection to goal failed. Returning closest node (Dist: {min_dist_to_goal:.2f}m)")
            return self._reconstruct_path(best_node)

        print("[RRT] Max iterations reached, path not found.")
        return []

    def _check_collision_trajectory(self, trajectory: List[State], grid_map: GridMap) -> bool:
        """辅助函数：检查一段轨迹是否碰撞"""
        if not trajectory:
            return False
        for s in trajectory:
            if self.collision_checker.check(self.vehicle, s, grid_map):
                return True
        return False

    def _get_random_sample(self, goal: State, grid_map: GridMap) -> State:
        """随机采样状态"""
        if random.random() < self.goal_sample_rate:
            return goal
        
        # 在地图范围内随机撒点
        phys_w = grid_map.width * grid_map.resolution
        phys_h = grid_map.height * grid_map.resolution
        
        rx = random.uniform(0, phys_w)
        ry = random.uniform(0, phys_h)
        
        return State(rx, ry, 0.0)

    def _get_nearest_node(self, node_list: List[RRTNode], rnd_state: State) -> RRTNode:
        """寻找最近节点 (欧氏距离)"""
        # 注意：对于非完整约束车辆，欧氏距离不是完美的度量，但在 RRT 中通常作为一种可接受的启发式
        dists = [(node.x - rnd_state.x)**2 + (node.y - rnd_state.y)**2 
                 for node in node_list]
        min_idx = dists.index(min(dists))
        return node_list[min_idx]

    def _steer(self, from_node: RRTNode, to_state: State) -> RRTNode:
        """
        核心生长函数：委托给 Vehicle 处理
        """
        # 这里的 max_dist 使用 self.step_size
        final_state, trajectory = self.vehicle.propagate_towards(
            start=from_node.state,
            target=to_state,
            max_dist=self.step_size
        )
        
        new_node = RRTNode(final_state)
        new_node.parent = from_node
        new_node.path_from_parent = trajectory
        
        return new_node

    def _check_collision(self, node: RRTNode, grid_map: GridMap) -> bool:
        """
        检查节点及其父路径是否碰撞
        """
        # 1. 检查最终状态
        if self.collision_checker.check(self.vehicle, node.state, grid_map):
            return True
            
        # 2. 检查生长过程中的轨迹点 (防止穿墙)
        # 如果 Vehicle 的 propagate_towards 返回的 trajectory 够密，这里就能保证安全
        if node.path_from_parent:
            for s in node.path_from_parent:
                if self.collision_checker.check(self.vehicle, s, grid_map):
                    return True
                 
        return False

    def _reconstruct_path(self, end_node: RRTNode) -> List[State]:
        """回溯路径，拼接详细轨迹"""
        path = []
        curr = end_node
        
        while curr is not None:
            # RRTNode 的 path_from_parent 通常是从 parent -> current 的顺序
            # 因为我们在回溯 (end -> start)，所以需要 reversed 这一小段，或者整体反转
            
            # 策略：先把整段 path_from_parent (parent -> curr) 倒序加入 (curr -> parent)
            if curr.path_from_parent:
                path.extend(reversed(curr.path_from_parent))
            else:
                # 起点节点没有 path_from_parent，直接加状态
                if not path or path[-1] != curr.state: # 避免重复
                    path.append(curr.state)
                
            curr = curr.parent
            
        # 现在 path 是从 Goal -> Start 的，翻转回 Start -> Goal
        return list(reversed(path))