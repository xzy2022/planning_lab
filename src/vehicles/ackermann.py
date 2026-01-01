# [实现] 具体类 B
# src/vehicles/ackermann.py
import math
from .base import VehicleBase, State  # 导入接口

class AckermannVehicle(VehicleBase):
    def __init__(self, wheelbase: float, max_steer: float):
        self.wheelbase = wheelbase
        self.max_steer = max_steer

    def get_shape(self) -> Tuple[float, float]:
        # 具体的阿克曼小车尺寸
        return (2.0, 4.5)

    def kinematic_propagate(self, start_state: State, control: tuple, dt: float) -> State:
        # 具体的阿克曼动力学积分公式 (Implementation)
        v, steering = control
        
        # 限制转角
        steering = max(min(steering, self.max_steer), -self.max_steer)

        new_theta = start_state.theta + (v / self.wheelbase) * math.tan(steering) * dt
        new_x = start_state.x + v * math.cos(new_theta) * dt
        new_y = start_state.y + v * math.sin(new_theta) * dt
        
        return State(new_x, new_y, new_theta)