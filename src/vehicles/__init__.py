# [入口] 负责暴露类，让外部调用更简洁

# src/vehicles/__init__.py

# 用户可以直接从 vehicles 包导入，而不需要知道具体文件名
from .base import VehicleBase, State
from .ackermann import AckermannVehicle
from .point_mass import PointMassVehicle 

# 定义对外暴露的列表
__all__ = ["VehicleBase", "State", "AckermannVehicle", "PointMassVehicle"]