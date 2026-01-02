# [实现] 具体类 B
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
        

    def get_collision_shapes(self, state: State):
        """
        CollisionChecker 调用此方法获取需要检测的圆
        将 Config 里的相对 offsets 变换到世界坐标
        """
        cos_theta = math.cos(state.theta)
        sin_theta = math.sin(state.theta)
        
        # 批量坐标变换
        cx_list = state.x + self.config.collision_offsets * cos_theta
        cy_list = state.y + self.config.collision_offsets * sin_theta
        
        return cx_list, cy_list, self.config.collision_radius

    def kinematic_propagate(self, start_state: State, control: tuple, dt: float) -> State:

        # 读取配置
        wb = self.config.wheelbase
        limit = self.config.max_steer  # 访问配置中的弧度值

        # 提取控制信号
        v, steering = control
        steering = max(min(steering, limit), -limit)
        
        # 计算新状态
        tan_steer = math.tan(steering)
        new_theta = start_state.theta + (v / wb) * tan_steer * dt
        new_theta = self.normalize_angle(new_theta)
        new_x = start_state.x + v * math.cos(new_theta) * dt
        new_y = start_state.y + v * math.sin(new_theta) * dt
        

        return State(new_x, new_y, new_theta)
    
    def get_visualization_polygon(self, state: State):
        """
        获取用于绘图的车辆轮廓多边形（世界坐标）
        """
        # 旋转矩阵 (2x2)
        c = math.cos(state.theta)
        s = math.sin(state.theta)
        
        # 1. 旋转: (N, 2) dot (2, 2) -> (N, 2)
        # 注意矩阵乘法顺序，outline_coords 是 (N, 2)
        # rotated = self.config.outline_coords @ rotation.T 
        # 或者手动展开写更清晰:
        
        # 这里 outline_coords 是 [x, y]，x向前，y向左
        # world_x = x * cos - y * sin + state_x
        # world_y = x * sin + y * cos + state_y
        
        # 利用 numpy 广播机制
        local_points = self.config.outline_coords
        world_points = np.zeros_like(local_points)
        
        world_points[:, 0] = local_points[:, 0] * c - local_points[:, 1] * s + state.x
        world_points[:, 1] = local_points[:, 0] * s + local_points[:, 1] * c + state.y
        
        return world_points