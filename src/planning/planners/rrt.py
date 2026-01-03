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
        self.state = state
        self.parent: Optional[RRTNode] = None
        self.path_from_parent: List[State] = [] # 记录从父节点到当前节点的微观轨迹

    @property
    def x(self): return self.state.x
    @property
    def y(self): return self.state.y

class RRTPlanner(PlannerBase):
    """
    基于采样和运动学的 RRT 规划器
    """
    def __init__(self, 
                 vehicle_model: VehicleBase,
                 collision_checker: CollisionChecker,
                 # RRT 特有参数
                 step_size: float = 2.0,       # 单次生长距离(米)或时间步长
                 max_iterations: int = 5000,   # 最大采样次数
                 goal_sample_rate: float = 0.1,# 目标偏置概率 (5%-10% 是经验值)
                 goal_threshold: float = 1.0   # 距离目标多近算到达
                 ):
        
        self.vehicle = vehicle_model
        self.collision_checker = collision_checker
        
        self.step_size = step_size
        self.max_iter = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.goal_threshold = goal_threshold

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

        print(f"[RRT] Start planning... Max Iter: {self.max_iter}")

        for i in range(self.max_iter):
            # 2. 采样 (Sample)
            # 有一定概率直接采样目标点 (Goal Bias)，加速收敛
            rnd_state = self._get_random_sample(goal, grid_map)

            # 3. 寻找最近邻 (Nearest)
            nearest_node = self._get_nearest_node(self.node_list, rnd_state)

            # 4. 生长 (Steer / Propagate)
            # 利用车辆运动学模型向前推演
            new_node = self._steer(nearest_node, rnd_state, self.step_size)

            # 5. 碰撞检测 (Collision Check)
            # 注意：不仅要检查终点，还要检查生长路径上的点是否碰撞
            if self._check_collision(new_node, grid_map):
                continue
            
            # 6. 添加到树
            self.node_list.append(new_node)
            
            # --- [Vis] 可视化调试 ---
            # 记录这次扩展，Plotter 可以画出一条线
            debugger.record_current_expansion(new_node.state)
            # 这里也可以调用 debugger 画出 nearest -> new_node 的连线

            # 7. 判断是否到达目标
            dist_to_goal = math.hypot(new_node.x - goal.x, new_node.y - goal.y)
            if dist_to_goal <= self.goal_threshold:
                print(f"[RRT] Goal reached at iter {i}!")
                return self._reconstruct_path(new_node)

        print("[RRT] Max iterations reached, path not found.")
        return []

    def _get_random_sample(self, goal: State, grid_map: GridMap) -> State:
        """随机采样状态"""
        if random.random() < self.goal_sample_rate:
            return goal
        
        # 在地图范围内随机撒点
        phys_w = grid_map.width * grid_map.resolution
        phys_h = grid_map.height * grid_map.resolution
        
        rx = random.uniform(0, phys_w)
        ry = random.uniform(0, phys_h)
        
        # 对于 RRT，采样的 theta 通常可以是随机的，
        # 或者在 _steer 阶段由几何关系计算，这里暂设为 0
        return State(rx, ry, 0.0)

    def _get_nearest_node(self, node_list: List[RRTNode], rnd_state: State) -> RRTNode:
        """
        寻找树中离采样点最近的节点
        注：这里用的是欧氏距离。对于 Ackermann 车辆，如果想更精确，
        可以使用 Reeds-Shepp 距离，但计算量会大增。
        """
        dists = [(node.x - rnd_state.x)**2 + (node.y - rnd_state.y)**2 
                 for node in node_list]
        min_idx = dists.index(min(dists))
        return node_list[min_idx]

    def _steer(self, from_node: RRTNode, to_state: State, extend_length: float) -> RRTNode:
        """
        核心生长函数：从 from_node 向 to_state 生长 extend_length 的距离
        """
        # 1. 计算期望的方向角
        dx = to_state.x - from_node.x
        dy = to_state.y - from_node.y
        target_yaw = math.atan2(dy, dx)
        
        # 2. 运动学推演
        # 我们需要计算应该输入的 steering angle。
        # 简单策略：Pure Pursuit 逻辑或 P控制器
        # 计算当前车头与目标方向的角度差
        curr_yaw = from_node.state.theta_rad
        diff_yaw = target_yaw - curr_yaw
        diff_yaw = (diff_yaw + math.pi) % (2 * math.pi) - math.pi # 归一化到 -pi ~ pi
        
        # 尝试以最大能力转向目标
        # 获取车辆的最大转角配置（如果有点话）
        max_steer = getattr(self.vehicle.config, 'max_steer', 1.0) # 默认为 1.0 rad (~57deg)
        steering = max(min(diff_yaw, max_steer), -max_steer)
        
        # 3. 使用 Vehicle 模型进行物理传播
        # 将 step_size 拆分为几个小步长进行积分，以获得更平滑的路径
        # 假设 extend_length 既代表距离，在恒定速度下也代表时间(v=1.0时)
        dt = 0.2
        num_steps = int(math.ceil(extend_length / dt)) # 简单处理：假设速度=1m/s
        
        curr_state = from_node.state
        path = []
        
        for _ in range(num_steps):
            # 只有全向车(PointMass)能直接横着走，Ackermann 必须遵循 steering
            # 这里统一用 kinematic_propagate，传入 (v, steering)
            # v = 1.0 m/s
            curr_state = self.vehicle.kinematic_propagate(curr_state, (1.0, steering), dt)
            path.append(curr_state)
            
        new_node = RRTNode(curr_state)
        new_node.parent = from_node
        new_node.path_from_parent = path # 存储这一段的详细轨迹
        
        return new_node

    def _check_collision(self, node: RRTNode, grid_map: GridMap) -> bool:
        """检查节点及其父路径是否碰撞"""
        # 1. 检查最终节点
        if self.collision_checker.check(self.vehicle, node.state, grid_map):
            return True
            
        # 2. 检查生长过程中的路径点 (防止穿墙)
        # RRT 的步长如果很大，中间必须采样检查
        for s in node.path_from_parent:
             if self.collision_checker.check(self.vehicle, s, grid_map):
                 return True
                 
        return False

    def _reconstruct_path(self, end_node: RRTNode) -> List[State]:
        """回溯路径"""
        path = []
        curr = end_node
        while curr is not None:
            # 注意：要把 path_from_parent 加入，才能得到平滑轨迹，而不仅仅是关键点
            # path_from_parent 是从 parent -> current 的顺序
            path.extend(reversed(curr.path_from_parent)) 
            # 如果 path_from_parent 为空（起点），加入 state
            if not curr.path_from_parent:
                path.append(curr.state)
                
            curr = curr.parent
            
        return list(reversed(path))