# src/map/generator.py
import numpy as np
import math
import random
from src.map.grid_map import GridMap
from src.vehicles.base import VehicleBase, State

class MapGenerator:
    """
    负责生成复杂地图的工厂类。
    解耦了 Map 和 Vehicle 的依赖关系。
    """
    
    @staticmethod
    def generate_feasible_map(
        grid_map: GridMap, 
        vehicle: VehicleBase, 
        start: State, 
        goal: State,
        obstacle_density: float = 0.4,
        max_steps: int = 2000
    ):
        """
        生成一个保证阿克曼车至少有一条可行路径的地图。
        原理：先生成高密度障碍物，然后控制车辆从起点强行开往终点，
        将轨迹覆盖的区域强制设为无障碍（推平）。
        """
        
        # 1. 先生成满屏的随机障碍物 (或者高密度)
        grid_map.generate_random_obstacles(density=obstacle_density)
        
        # 确保起点和终点本身是空的
        MapGenerator._clear_area(grid_map, start.x, start.y, radius=2.0)
        MapGenerator._clear_area(grid_map, goal.x, goal.y, radius=2.0)
        
        # 2. 模拟车辆“推土机”式运行
        current_state = start
        dt = 0.5 # 仿真步长
        
        for _ in range(max_steps):
            # 简单策略：朝向目标点的方向行驶 (P控制)
            dx = goal.x - current_state.x
            dy = goal.y - current_state.y
            target_yaw = math.atan2(dy, dx)
            
            # 计算此时需要的转向角 (简单的 P 控制器)
            diff_yaw = target_yaw - current_state.theta
            # 归一化角度到 -pi ~ pi
            diff_yaw = (diff_yaw + math.pi) % (2 * math.pi) - math.pi
            
            steer = max(min(diff_yaw, 0.6), -0.6) # 假定最大转角 0.6
            velocity = 1.0 # 恒定速度前进
            
            # 3. 动力学推演 (使用车辆自身的模型)
            next_state = vehicle.kinematic_propagate(
                current_state, (velocity, steer), dt
            )
            
            # 4. [关键] "推平"：清除车辆当前位置周围的障碍物
            # 这里简单地清除车辆外接圆半径范围内的障碍
            # 也可以更精细地根据 get_shape() 清除矩形区域
            width, length = vehicle.get_shape()
            clear_radius = max(width, length) / 2.0 + 0.5 # 留一点余量
            
            MapGenerator._clear_area(grid_map, next_state.x, next_state.y, clear_radius)
            
            current_state = next_state
            
            # 判断是否到达附近
            if math.hypot(dx, dy) < 2.0:
                print("Generator: Feasible path created!")
                break
                
    @staticmethod
    def _clear_area(grid_map: GridMap, center_x: float, center_y: float, radius: float):
        """辅助函数：将圆形区域内的网格设为 0 (Free)"""
        # 将物理坐标转为网格索引
        cx_idx = int(center_x / grid_map.resolution)
        cy_idx = int(center_y / grid_map.resolution)
        r_grids = int(radius / grid_map.resolution)
        
        # 遍历圆形包围盒内的网格
        for y in range(cy_idx - r_grids, cy_idx + r_grids + 1):
            for x in range(cx_idx - r_grids, cx_idx + r_grids + 1):
                # 检查边界
                if 0 <= x < grid_map.width and 0 <= y < grid_map.height:
                    # 检查是否在圆内 (避免清除方形)
                    if (x - cx_idx)**2 + (y - cy_idx)**2 <= r_grids**2:
                        grid_map.data[y, x] = 0 # 设为无障碍