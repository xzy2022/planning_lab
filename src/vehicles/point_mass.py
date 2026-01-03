# src/vehicles/point_mass.py
import numpy as np
from typing import Tuple
from .base import VehicleBase, State
from .config import PointMassConfig
import math

class PointMassVehicle(VehicleBase):
    """
    质点/全向移动车辆模型实现
    
    特点：
    1. 运动学：全向移动 (Holonomic)，直接接受 (vx, vy) 控制。
    2. 几何：通常视为圆形，但在网格地图中占据特定大小。
    3. 旋转：本身不旋转 (theta 保持不变或无意义)，始终保持轴对齐 (AABB)。
    """
    
    def __init__(self, config: PointMassConfig):
        super().__init__(config)
        self.config: PointMassConfig = config

    def get_bounding_circle(self, state: State) -> Tuple[float, float, float]:
        """
        [粗检测] 返回包围圆
        对于质点模型，圆心即自身坐标 (x, y)。
        """
        # 半径 = 配置半径 + 安全余量
        total_radius = self.config.bounding_radius + self.config.safe_margin
        return state.x, state.y, total_radius

    def get_collision_circles(self, state: State) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        [精检测 - 多圆]
        质点模型通常只需要一个圆即可完美覆盖。
        """
        cx_list = np.array([state.x])
        cy_list = np.array([state.y])
        total_radius = self.config.bounding_radius + self.config.safe_margin
        return cx_list, cy_list, total_radius

    def get_collision_polygon(self, state: State) -> np.ndarray:
        """
        [精检测 - 多边形]
        为了兼容 SAT 碰撞检测或可视化，生成一个以 (x,y) 为中心的正方形/矩形。
        """
        return self.transform_points(self.config.outline_coords, state)

    def kinematic_propagate(self, start_state: State, control: tuple, dt: float) -> State:
        """
        物理推演
        :param control: (vx, vy) 速度矢量
        """
        vx, vy = control
        
        # 1. 简单的欧拉积分更新位置
        new_x = start_state.x + vx * dt
        new_y = start_state.y + vy * dt
        
        # 2. 角度处理
        # PointMass 不旋转，保持原状；或者根据需要设为 0
        new_theta = start_state.theta_rad 
        
        return State(new_x, new_y, new_theta)
    
    def geometric_step(self, start: State, displacement: tuple) -> State:
        """
        基于位移的瞬时移动。
        适用于 A* (4向/8向) 或 Geometric RRT (直接连线)。
        
        Args:
            start: 起点
            displacement: (dx, dy) 期望移动的向量
        """
        dx, dy = displacement
        
        # 限制最大步长 (可选，防止穿墙)
        # dist = math.hypot(dx, dy)
        # if dist > self.config.max_step: ...
        
        new_x = start.x + dx
        new_y = start.y + dy
        
        return State(new_x, new_y, start.theta_rad)
    
    def propagate_towards(self, start: State, target: State, max_dist: float) -> tuple[State, list[State]]:
        """
        RRT 接口：直接向目标连线
        """
        # 1. 计算向量
        dx = target.x - start.x
        dy = target.y - start.y
        dist = math.hypot(dx, dy)
        
        # 2. 截断距离
        step = min(dist, max_dist)
        
        # 3. 归一化方向
        if dist > 1e-6:
            dx = (dx / dist) * step
            dy = (dy / dist) * step
        else:
            dx, dy = 0, 0
            
        # 4. 复用你刚才写的 geometric_step !
        final_state = self.geometric_step(start, (dx, dy))
        
        # 5. 对于几何直线，轨迹只要两头即可 (或者你可以生成中间插值点)
        trajectory = [final_state] 
        
        return final_state, trajectory

    def get_visualization_polygon(self, state: State) -> np.ndarray:
        """
        [可视化]
        直接复用碰撞多边形。
        """
        return self.get_collision_polygon(state)
        