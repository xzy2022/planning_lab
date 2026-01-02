# src/map/generator.py
import numpy as np
import math
import random
from src.map.grid_map import GridMap
from src.types import State
from src.vehicles.base import VehicleBase

class MapGenerator:
    """
    负责生成复杂地图的工厂类。
    解耦了 Map 和 Vehicle 的依赖关系。
    """

    @staticmethod
    def generate_random_obstacles(grid_map: GridMap, density: float = 0.1, seed: int = None):
        """
        [移植] 静态方法：直接修改传入的 grid_map
        """
        if seed is not None:
            np.random.seed(seed)
            
        width = grid_map.width
        height = grid_map.height
        
        # 生成随机 mask
        random_mask = np.random.rand(height, width) < density
        
        # 修改 grid_map 的数据
        grid_map.data[random_mask] = 1
        
        # 强制设置四面围墙
        grid_map.data[0, :] = 1
        grid_map.data[-1, :] = 1
        grid_map.data[:, 0] = 1
        grid_map.data[:, -1] = 1

    @staticmethod
    def inflate_obstacles(grid_map: GridMap, radius_grids: int):
        """
        [移植] 障碍物膨胀算法
        """
        if radius_grids <= 0:
            return

        # 获取当前数据的引用
        data = grid_map.data
        rows, cols = data.shape
        
        # 找到所有障碍物点
        obs_y, obs_x = np.where(data == 1)
        
        # 创建临时副本用于计算膨胀结果 (避免原地修改导致无限生长)
        inflated_grid = data.copy()
        
        for y, x in zip(obs_y, obs_x):
            y_min = max(0, y - radius_grids)
            y_max = min(rows, y + radius_grids + 1)
            x_min = max(0, x - radius_grids)
            x_max = min(cols, x + radius_grids + 1)
            
            inflated_grid[y_min:y_max, x_min:x_max] = 1
            
        # 将结果写回 grid_map
        # 注意：必须使用 [:] 全量切片赋值，才能修改原对象的内容，
        # 如果用 grid_map.data = inflated_grid 可能会因为没有 setter 而失败或只是修改了局部变量
        grid_map.data[:] = inflated_grid[:]

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
        生成可行地图的主流程
        """
        # 1. 调用自身的静态方法生成随机障碍
        # [修改] 不再调用 grid_map.generate... 而是调自己的 MapGenerator.generate...
        MapGenerator.generate_random_obstacles(grid_map, density=obstacle_density)
        
        # 确保起点终点无障碍
        MapGenerator._clear_area(grid_map, start.x, start.y, radius=2.0)
        MapGenerator._clear_area(grid_map, goal.x, goal.y, radius=2.0)
        
        # 2. 模拟推土机
        current_state = start
        dt = 0.5 
        max_steer = getattr(vehicle.config, 'max_steer', 0.6)

        for _ in range(max_steps):
            dx = goal.x - current_state.x
            dy = goal.y - current_state.y
            
            # 判断到达
            if math.hypot(dx, dy) < 2.0:
                print("Generator: Feasible path created!")
                break

            target_yaw = math.atan2(dy, dx)
            diff_yaw = target_yaw - current_state.theta_rad
            diff_yaw = (diff_yaw + math.pi) % (2 * math.pi) - math.pi
            
            steer = max(min(diff_yaw, max_steer), -max_steer)
            velocity = 1.0 
            
            next_state = vehicle.kinematic_propagate(current_state, (velocity, steer), dt)
            
            bx, by, b_radius = vehicle.get_bounding_circle(next_state)
            MapGenerator._clear_area(grid_map, bx, by, b_radius)
            
            current_state = next_state
                
    @staticmethod
    def _clear_area(grid_map: GridMap, center_x: float, center_y: float, radius: float):
        """辅助函数：将圆形区域内的网格设为 0 (Free)"""
        
        # [修正] 1. 使用 GridMap 提供的标准接口进行坐标转换
        cx_idx, cy_idx = grid_map.world_to_grid(center_x, center_y)
        
        # [修正] 2. 计算半径对应的格数 (向上取整以确保覆盖边缘)
        r_grids = int(math.ceil(radius / grid_map.resolution))
        
        # 遍历圆形包围盒内的网格
        for y in range(cy_idx - r_grids, cy_idx + r_grids + 1):
            for x in range(cx_idx - r_grids, cx_idx + r_grids + 1):
                # 使用 GridMap 的内部边界检查 (或者直接判断)
                if 0 <= x < grid_map.width and 0 <= y < grid_map.height:
                    # 检查是否在圆内 (避免清除方形)
                    if (x - cx_idx)**2 + (y - cy_idx)**2 <= r_grids**2:
                        grid_map.data[y, x] = 0 # 设为无障碍