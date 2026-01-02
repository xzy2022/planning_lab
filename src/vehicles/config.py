# [配置] 该模块独有的配置数据类
from dataclasses import dataclass

@dataclass
class VehicleConfig:
    """所有车辆通用的配置"""
    collision_radius: float = 1.0  # 用于粗略的圆形碰撞检测
    max_velocity: float = 2.0
    
@dataclass
class AckermannConfig(VehicleConfig):
    """阿克曼模型特有配置"""
    wheelbase: float = 2.5
    max_steer: float = 0.6
    width: float = 2.0  # 具体的矩形尺寸
    length: float = 4.5

@dataclass
class PointMassConfig(VehicleConfig):
    """质点模型特有配置"""
    # 质点可能不需要额外配置，但保留扩展性
    pass