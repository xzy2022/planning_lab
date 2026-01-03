# [配置] 该模块独有的配置数据类
from dataclasses import dataclass, field
import numpy as np
import math

@dataclass
class VehicleConfig:
    """所有车辆通用的配置"""
    safe_margin: float = 0.1  # 碰撞检测的安全余量
    max_velocity: float = 2.0
    
@dataclass
class AckermannConfig(VehicleConfig):
    """
    阿克曼车辆物理参数配置
    借鉴了工业级设计：几何参数与碰撞参数分离，但在初始化时预计算
    """
    # --- 1. 基础几何参数 (核心) ---
    wheelbase: float = 2.5       # [m] 轴距
    width: float = 2.0           # [m] 车宽
    front_hang: float = 0.9      # [m] 前悬 (前轴中心到车头)
    rear_hang: float = 0.9       # [m] 后悬 (后轴中心到车尾)
    
    # --- 2. 运动学限制 ---
    max_steer_deg: float = 35.0  # [deg] 最大转向角
    
    # --- 3. 派生属性 (自动计算，外部只读) ---
    max_steer: float = field(init=False)
    collision_radius: float = field(init=False)      # 单个小圆的半径
    collision_offsets: np.ndarray = field(init=False)# 小圆圆心相对于后轴中心的距离列表
    outline_coords: np.ndarray = field(init=False)   # 用于绘图的矩形框坐标(4x2)

    def __post_init__(self):
        # A. 角度转弧度
        self.max_steer = math.radians(self.max_steer_deg)
        
        # B. 预计算多圆覆盖参数 (这是属于车辆的几何特征)
        # 策略：沿车身纵轴分布多个圆
        total_length = self.wheelbase + self.front_hang + self.rear_hang
        
        # 半径 = 车宽的一半 + 安全余量 (确保覆盖车角)
        # 借鉴：稍微大一点点以覆盖矩形对角线
        self.collision_radius = (self.width / 2.0) * 1.1 + self.safe_margin
        
        # 确定圆的数量：保证圆之间重叠足够，不漏掉缝隙
        # 经验公式：每隔 radius 的距离放一个圆
        num_circles = int(np.ceil(total_length / (self.collision_radius * 0.8)))
        num_circles = max(num_circles, 3) # 至少3个圆
        
        # 生成偏移量 (相对于后轴中心 (0,0))
        # 后轴中心的X是0。车尾是 -rear_hang，车头是 +wheelbase + front_hang
        start_x = -self.rear_hang + self.collision_radius * 0.5
        end_x = (self.wheelbase + self.front_hang) - self.collision_radius * 0.5
        
        self.collision_offsets = np.linspace(start_x, end_x, num_circles)

        # C. 预计算绘图轮廓 (相对于后轴中心)
        # 顺时针: 右前 -> 右后 -> 左后 -> 左前 (根据matplotlib或绘图库习惯调整)
        # x轴向前，y轴向左
        front_x = self.wheelbase + self.front_hang
        rear_x = -self.rear_hang
        left_y = self.width / 2.0
        right_y = -self.width / 2.0
        
        self.outline_coords = np.array([
            [front_x, right_y],
            [rear_x,  right_y],
            [rear_x,  left_y],
            [front_x, left_y],
            [front_x, right_y] # 闭合
        ])

@dataclass
class AckermannConfig(VehicleConfig):
    """
    阿克曼车辆物理参数配置
    借鉴了工业级设计：几何参数与碰撞参数分离，但在初始化时预计算
    """
    # --- 1. 基础几何参数 (核心) ---
    wheelbase: float = 2.5       # [m] 轴距
    width: float = 2.0           # [m] 车宽
    front_hang: float = 0.9      # [m] 前悬 (前轴中心到车头)
    rear_hang: float = 0.9       # [m] 后悬 (后轴中心到车尾)
    
    # --- 2. 运动学限制 ---
    max_steer_deg: float = 35.0  # [deg] 最大转向角
    max_velocity: float = 2.0    # [m/s] 最大速度
    
    # --- 3. 派生属性 (自动计算，外部只读) ---
    max_steer: float = field(init=False)
    collision_radius: float = field(init=False)      # 单个小圆的半径
    collision_offsets: np.ndarray = field(init=False)# 小圆圆心相对于后轴中心的距离列表
    outline_coords: np.ndarray = field(init=False)   # 用于绘图的矩形框坐标(4x2)
    bounding_radius: float = field(init=False)      # 最小外接圆半径 (不含安全余量)
    bounding_offset: np.ndarray = field(init=False) # 最小外接圆圆心相对于后轴的坐标 [x, y]

    def __post_init__(self):
        # A. 角度转弧度
        self.max_steer = math.radians(self.max_steer_deg)
        
        # B. 预计算多圆覆盖参数 (这是属于车辆的几何特征)
        # 策略：沿车身纵轴分布多个圆
        total_length = self.wheelbase + self.front_hang + self.rear_hang
        
        # 半径 = 车宽的一半 + 安全余量 (确保覆盖车角)
        # 借鉴：稍微大一点点以覆盖矩形对角线
        self.collision_radius = (self.width / 2.0) * 1.1 + self.safe_margin
        
        # 确定圆的数量：保证圆之间重叠足够，不漏掉缝隙
        # 经验公式：每隔 radius 的距离放一个圆
        num_circles = int(np.ceil(total_length / (self.collision_radius * 0.8)))
        num_circles = max(num_circles, 3) # 至少3个圆
        
        # 生成偏移量 (相对于后轴中心 (0,0))
        # 后轴中心的X是0。车尾是 -rear_hang，车头是 +wheelbase + front_hang
        start_x = -self.rear_hang + self.collision_radius * 0.5
        end_x = (self.wheelbase + self.front_hang) - self.collision_radius * 0.5
        
        self.collision_offsets = np.linspace(start_x, end_x, num_circles)

        # C. 预计算绘图轮廓 (相对于后轴中心)
        # 顺时针: 右前 -> 右后 -> 左后 -> 左前 (根据matplotlib或绘图库习惯调整)
        # x轴向前，y轴向左
        front_x = self.wheelbase + self.front_hang
        rear_x = -self.rear_hang
        left_y = self.width / 2.0
        right_y = -self.width / 2.0
        
        self.outline_coords = np.array([
            [front_x, right_y],
            [rear_x,  right_y],
            [rear_x,  left_y],
            [front_x, left_y],
            [front_x, right_y] # 闭合
        ])

        # ---  通用最小外接圆计算 (AABB法) ---
        # 1. 计算局部坐标系下的 AABB 中心 (对于矩形车身，这就是几何中心)
        min_xy = np.min(self.outline_coords, axis=0)
        max_xy = np.max(self.outline_coords, axis=0)
        center_local = (min_xy + max_xy) / 2.0
        
        # 2. 计算覆盖所有顶点的最小半径
        # 向量化计算所有点到中心的距离
        dists = np.linalg.norm(self.outline_coords - center_local, axis=1)
        geom_radius = np.max(dists)
        
        # 3. 存入属性
        self.bounding_offset = center_local
        self.bounding_radius = geom_radius


@dataclass
class PointMassConfig(VehicleConfig):
    """
    质点/全向小车模型配置
    """
    # --- 1. 几何参数 ---
    width: float = 1.0           # [m] 车宽 (用于碰撞矩形)
    length: float = 1.0          # [m] 车长
    
    # --- 2. 运动参数 暂无---

    
    # --- 3. 派生属性 (自动计算) ---
    bounding_radius: float = field(init=False)      # 外接圆半径
    bounding_offset: np.ndarray = field(init=False) # 外接圆偏移 (通常为 0,0)
    outline_coords: np.ndarray = field(init=False)  # 矩形轮廓

    def __post_init__(self):
        # A. 预计算轮廓 (以中心为原点的矩形)
        # 顺时针: 右前 -> 右后 -> 左后 -> 左前
        dx = self.length / 2.0
        dy = self.width / 2.0
        
        self.outline_coords = np.array([
            [dx, -dy],
            [-dx, -dy],
            [-dx, dy],
            [dx, dy],
            [dx, -dy] # 闭合
        ])
        
        # B. 预计算外接圆 (AABB 中心即为几何中心)
        self.bounding_offset = np.array([0.0, 0.0])
        # 半径为矩形对角线的一半。这里没有考虑安全余量。
        self.bounding_radius = math.hypot(dx, dy)