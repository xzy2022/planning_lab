# src/vehicles/ackermann.py
import math
from .base import VehicleBase, State  # 导入接口
from .config import AckermannConfig  # 导入配置类
from typing import Tuple
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
    
    def get_visualization_polygon(self, state: State) -> np.ndarray:
            """
            [表现层] 返回用于给人看的多边形
            通常在调试算法时，我们希望看到算法眼中的'碰撞盒'，
            所以默认返回 get_collision_polygon 的结果是非常合理的工程实践。
            """
            # 策略 1: 直接复用碰撞形状（最常见于 Planning Debug）
            return self.get_collision_polygon(state)