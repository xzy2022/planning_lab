# src/vehicles/ackermann.py
import math
from .base import VehicleBase, State  # 导入接口
from .config import AckermannConfig  # 导入配置类
from typing import Tuple, List 
import numpy as np

class AckermannVehicle(VehicleBase):
    def __init__(self, config: AckermannConfig):
        super().__init__(config)
        self.config: AckermannConfig = config
        

    def get_bounding_circle(self, state: State) -> Tuple[float, float, float]:
        """
        [精确接口]
        返回世界坐标系下的最小外接圆 (x, y, radius)。
        return:
        center_x, center_y: 圆心坐标相对于小车原点(后轴中心)的偏移
        radius: 圆的半径(考虑安全裕量)
        """
        # 1. 将局部偏移量转换到世界坐标系
        # 利用基类的 transform_points 或手动旋转
        # 这里的 offset 是一个点，看作向量旋转
        c = math.cos(state.theta_rad)
        s = math.sin(state.theta_rad)
        
        ox, oy = self.config.bounding_offset
        
        # 旋转 + 平移
        center_x = ox * c - oy * s + state.x
        center_y = ox * s + oy * c + state.y
        
        return center_x, center_y, self.config.bounding_radius + self.config.safe_margin

    def get_collision_circles(self, state: State):
        # 你的多圆实现
        cos_theta = math.cos(state.theta_rad)
        sin_theta = math.sin(state.theta_rad)
        cx_list = state.x + self.config.collision_offsets * cos_theta
        cy_list = state.y + self.config.collision_offsets * sin_theta
        return cx_list, cy_list, self.config.collision_radius

    def get_collision_polygon(self, state: State) -> np.ndarray:
            """
            [物理层] 返回用于碰撞检测的多边形
            在当前配置下，它直接使用了车身轮廓。
            但未来如果你想给碰撞加个 '膨胀系数'，就在这里改，而不会影响画图。
            """
            # 假设 config 里有一个专门用于碰撞的 simplified_polygon，或者直接用 outline
            return self.transform_points(self.config.outline_coords, state)

    def kinematic_propagate(self, start_state: State, control: tuple, dt: float) -> State:

        # 读取配置
        wb = self.config.wheelbase
        limit = self.config.max_steer  # 访问配置中的弧度值

        # 提取控制信号
        v, steering = control
        steering = max(min(steering, limit), -limit)
        
        # 计算新状态
        tan_steer = math.tan(steering)
        new_theta = start_state.theta_rad + (v / wb) * tan_steer * dt
        new_theta = self.normalize_angle(new_theta)
        new_x = start_state.x + v * math.cos(new_theta) * dt
        new_y = start_state.y + v * math.sin(new_theta) * dt
        

        return State(new_x, new_y, new_theta)
    

    def propagate_towards(self, start: State, target: State, max_dist: float) -> Tuple[State, List[State]]:
        """
        [实现接口] 尝试从 start 向 target 移动。
        对于阿克曼小车，这意味着计算指向 target 的方向盘转角，并向前模拟一段距离。
        """
        # 1. 计算相对位置
        dx = target.x - start.x
        dy = target.y - start.y
        dist_to_target = math.hypot(dx, dy)

        # 如果距离极小，认为已经到达
        if dist_to_target < 1e-3:
            return start, [start]

        # 2. 决定移动步长 (RRT Extend 逻辑)
        # 我们不能超过 max_dist，也不能超过实际到目标的距离
        move_dist = min(dist_to_target, max_dist)

        # 3. 决定控制量 (v, steering)
        # 计算目标点相对于当前车头的角度偏差
        target_angle = math.atan2(dy, dx)
        angle_diff = self.normalize_angle(target_angle - start.theta_rad)

        # 简单的 Steering 策略：
        # 将方向盘转角设置为角度偏差，但必须限制在 max_steer 范围内 (Clamp)
        # 注意：这是一种贪心策略(Greedy)，假设朝着目标方向打方向盘就能靠近。
        limit = self.config.max_steer
        steering = max(min(angle_diff, limit), -limit)
        
        # 速度：默认使用配置中的最大速度，或者设为定值 1.0 m/s
        # 假设 config 中有 max_v，如果没有建议在 Config 里加上，或暂时硬编码
        v = getattr(self.config, 'max_velocity', 2.0) 

        # 4. 离散化积分生成轨迹
        dt = 0.1  # 积分时间步长 (秒)
        # 计算需要的总时间
        total_time = move_dist / v
        # 计算步数 (向上取整)
        steps = int(math.ceil(total_time / dt))
        
        trajectory = [start]
        current_state = start

        for _ in range(steps):
            # 复用 kinematic_propagate 进行单步推演
            current_state = self.kinematic_propagate(current_state, (v, steering), dt)
            trajectory.append(current_state)

        return current_state, trajectory
    
    def get_visualization_polygon(self, state: State) -> np.ndarray:
            """
            [表现层] 返回用于给人看的多边形
            通常在调试算法时，我们希望看到算法眼中的'碰撞盒'，
            所以默认返回 get_collision_polygon 的结果是非常合理的工程实践。
            """
            # 策略 1: 直接复用碰撞形状（最常见于 Planning Debug）
            return self.get_collision_polygon(state)