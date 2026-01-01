# src/vehicles/point_mass.py

from .base import VehicleBase, State

class PointMassVehicle(VehicleBase):
    """
    类质点车辆模型 (Point Mass Model)
    
    特点：
    1. 不考虑动力学和非完整约束（无转弯半径限制）。
    2. 可以全向移动（Holonomic），直接改变 x, y 坐标实现平移或斜向运动。
    3. 保留形状用于碰撞检测。
    """
    def update_state(self, vx, vy, dt):
        # 简单的位置更新，直接平移
        self.state.x += vx * dt
        self.state.y += vy * dt
        # 如果需要朝向跟随速度方向（可选）：
        # import math
        # if vx != 0 or vy != 0:
        #     self.state.yaw = math.atan2(vy, vx)